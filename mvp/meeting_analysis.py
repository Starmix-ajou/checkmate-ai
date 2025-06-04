import logging
import os
from collections import defaultdict
from typing import List

import torch
from dotenv import load_dotenv
from gpt_utils import extract_json_from_gpt_response
from huggingface_hub import login
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mongodb_setting import get_epic_collection
from openai import AsyncOpenAI
from project_member_utils import get_project_members
from redis_setting import load_from_redis, save_to_redis
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          TokenClassificationPipeline)

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
login(token=HUGGINGFACE_API_KEY)


'''
Token은 BIO 형식 기반으로 처리. BIO 형식은 각 토큰에 대해 태그 부여 방식을 정의한 것.

B: Begin, I: Inside, O: Outside
B-PER: 시작 토큰, I-PER: 중간 토큰, O: 그 외 토큰
'''


specify_model_name = "monologg/koelectra-base-v3-naver-ner"
'''
스펙 짧게 정리: 
    - koelectra-base-v3 (한국어 ELECTRA 변형)이 기반 모델
    - Input: 512 tokens
    - Class: 41개 NER tag에 대해 학습함
    - tokenizer 특징: 공백, 단어 경계 무시 && Sentence 단위로 분리함
    - F1-SCORE: 92.4
    - # of parameters: 110M
    - 속도: 빠름
'''

tokenizer = AutoTokenizer.from_pretrained(specify_model_name)
model_for_ner = AutoModelForTokenClassification.from_pretrained(specify_model_name)

async def create_action_items_finetuned(content: str):
    logger.info(f"🔍 회의 액션 아이템 생성 시작")
    
    # 모델의 입력 token 단위가 512개이므로, 이를 맞추기 위해 문단 단위로 텍스트 전처리
    paragraphs = content.split("\n\n")
    entities = []
    
    # 각 문단별로 처리
    for paragraph in paragraphs:
        if not paragraph.strip():   # 비어 있는 문단은 생략
            continue
        
        # 문단을 토큰화
        inputs = tokenizer(
            paragraph,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # 모델의 입력 token 단위가 512개이므로, 이를 명시적으로 맞춰서 처리
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        offset_mapping = inputs.pop("offset_mapping")[0] # offset mapping을 해야 이후에 토큰에 대응되는 문자열을 복원할 수 있음
        
        with torch.no_grad():
            outputs = model_for_ner(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].tolist()   # list로 묶어서 차원 축소
        input_ids = inputs["input_ids"][0]
        id2label = model_for_ner.config.id2label
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
        current_entity = {"type": None, "text": "", "start": None}
    
        for i, pred in enumerate(predictions):
            label = id2label[pred]
            start, end = offset_mapping[i].tolist()
            text_span = paragraph[start:end]
        
            if "-" in label:
                entity_type, tag = label.split("-")
            else:
                entity_type, tag = label, "0"
                
            if tag == "B":
                # 이전 entity 저장
                if current_entity["text"]:
                    entities.append(current_entity.copy())
                current_entity = {
                    "type": entity_type,
                    "text": text_span,
                    "start": start,
                }
            elif tag == "I" and current_entity["type"] == entity_type:
                current_entity["text"] += text_span
            else:
                # Outside 태그 / type 불일치
                if current_entity["text"]:
                    entities.append(current_entity.copy())
                    current_entity = {"type": None, "text": "", "start": None}
                    
        if current_entity["text"]:
            # 마지막 entity 처리
            entities.append(current_entity.copy())
    
    # 디버깅용 출력
    print("=== Extracted Entities ===")
    for ent in entities:
        print(ent)
        
    # task를 중심으로 assignee, enddate mapping
    action_items = []
    tasks = [e for e in entities if e["type"] in {"EVT", "TRM"}]
    assignees = [e for e in entities if e["type"] == "PER"]
    enddates = [e for e in entities if e["type"] == "DAT"]
    
    for task in tasks:
        closest_assignee = min(assignees, key=lambda x: abs(x["start"] - task["start"]), default=None)
        closest_enddate = min(enddates, key=lambda x: abs(x["start"] - task["start"]), default=None)
        
        action_items.append({
            "task": task["text"],
            "assignee": closest_assignee["text"] if closest_assignee else "",
            "enddate": closest_enddate["text"] if closest_enddate else "",
        })
        
    ''' 
    다음과 같은 형태로 action_items가 반환되면 성공
    [
        {
            "task": "보고서 제출",
            "assignee": "김승연",
            "enddate": "2024년 10월 1일"
        },
        {
            "task": "자료 정리",
            "assignee": "",
            "enddate": ""
        },
    ]
    '''
    print(f"최종 처리된 action_items: {action_items}")
    
    return action_items

### ============================== Current: Summary & Action Items Extraction ============================== ###

async def create_action_items_gpt(content: str):
    logger.info(f"🔍 회의 액션 아이템 생성 시작")
    
    action_items_prompt = ChatPromptTemplate.from_template("""
    당신은 회의록으로부터 액션 아이템을 정리해주는 AI 비서입니다. 당신의 주요 언어는 한국어입니다.
    회의록 {content}를 분석하여 다음 세 가지 요소를 포함한 액션 아이템을 정의해 주세요:
    1. 액션 아이템 내용
    2. 담당자
    3. 마감 기한
    
    세 가지 요소 중 회의록에 정보가 없는 요소는 null을 반환해 주세요.
    
    결과를 다음과 같은 형식으로 반환해 주세요:
    {{
        "actionItems": [
            {{
                "task": "string",
                "assignee": "string" | null,
                "enddate": "string" | null
            }},
            ...
        ]
    }}
    """)
    messages = action_items_prompt.format(content=content)
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.0,
    )
    response = await llm.ainvoke(messages)
    try:
        content = response.content
        try:
            gpt_result = extract_json_from_gpt_response(content)
        except Exception as e:
            logger.error(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True)
            raise Exception(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True) from e
    except Exception as e:
        logger.error(f"GPT API 처리 중 오류 발생: {e}", exc_info=True)
        raise Exception(f"GPT API 처리 중 오류 발생: {str(e)}", exc_info=True) from e
    
    action_items = gpt_result["actionItems"]
    print(f"생성된 액션 아이템: {action_items}")
    
    return action_items

async def create_summary(title: str, content: str, project_id: str):
    '''
    title: 사용자가 제목으로 회의록을 대표하는 내용을 입력한다고 가정 -> 요약의 첫 번째 뼈대로 사용
    content: Markdown 형태로 문서가 제공됨
    '''
    logger.info(f"🔍 회의 요약 생성 시작")
    
    print(f"회의 제목: {title}")
    meeting_summary_prompt = ChatPromptTemplate.from_template("""
    당신은 회의록에서 중요한 대화 내용을 정리해 주는 AI 비서입니다. 당신의 주요 언어는 한국어입니다. 정리한 내용은 반드시 Markdown 형식으로 반환해 주세요.
    당신의 업무는 회의 제목인 {title}을 바탕으로 회의록 {content}를 분석하여 중요한 대화 내용을 정리하는 것입니다.
    {title}은 회의의 제목으로서 회의록에서 논의되는 내용을 대표하는 것으로 간주합니다. 따라서 회의록의 내용을 분석할 때 {title}을 적극적으로 참조하세요.
    Heading이 있는 경우, 요약 과정에서도 Heading 레벨을 유지해 주세요. 예를 들어, 회의록에 "## 회의 요약"이라는 내용이 있는 경우, 요약 결과에도 "## 회의 요약"이 존재해야 합니다.
    회의록에서 중요한 대화 내용을 정리해 주세요. 중요한 대화 내용은 다음과 같습니다:
    - 회의 안건
    - 안건에 대한 논의 결과
    - 다음 회의 안건
    - 중요한 피드백 및 의견
    
    현재 프로젝트에 참여 중인 멤버들의 정보는 다음과 같습니다:
    {project_members}
    멤버들 중에서 특정 이름을 가진 발화자가 있는 경우 발화자의 이름과 발화 내용을 하나의 문장으로 묶어서 정리해 주세요.
    
    결과를 다음과 같은 형식으로 반환하세요:
    {{
        "summary": "Markdown 형식의 회의 요약 string",
    }}
    """)
    
    project_members = await get_project_members(project_id)
    
    messages = meeting_summary_prompt.format(
        title=title, 
        content=content,
        project_members=project_members)
    
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.1,
    )
    response = await llm.ainvoke(messages)
    
    try:
        content = response.content
        try:
            gpt_result = extract_json_from_gpt_response(content)
        except Exception as e:
            logger.error(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True)
            raise Exception(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True) from e
        
    except Exception as e:
        logger.error(f"GPT API 처리 중 오류 발생: {e}", exc_info=True)
        raise Exception(f"GPT API 처리 중 오류 발생: {str(e)}", exc_info=True) from e
    
    summary = gpt_result["summary"]
    print(f"회의 요약 결과: {summary}")
    
    return summary


async def analyze_meeting_document(meeting_id: str, title: str, content: str, project_id: str):
    global action_items, summary
    # 초기화 필수!
    action_items = None
    summary = None
    
    action_items = await create_action_items_gpt(content)
    try:
        await save_to_redis(f"action_items:{str(project_id)}", action_items)
    except Exception as e:
        logger.error(f"action_items 저장 중 오류 발생: {str(e)}", exc_info=True)
        raise Exception(f"action_items 저장 중 오류 발생: {str(e)}", exc_info=True) from e
    
    summary = await create_summary(title, content, project_id)
    
    # action_items에서 task들만 추출
    task_list = [item["task"] for item in action_items]
    
    response = {
        "summary": summary,
        "actionItems": task_list,
    }
    logger.info(f"구성된 response: {response}")
    return response


async def convert_action_items_to_tasks(actionItems: List[str], project_id: str):
    try:
        redis_action_items = await load_from_redis(f"action_items:{str(project_id)}")
    except Exception as e:
        logger.error(f"action_items 로드 중 오류 발생: {str(e)}", exc_info=True)
        raise Exception(f"action_items 로드 중 오류 발생: {str(e)}", exc_info=True) from e
    
    assert redis_action_items is not None, "정의되어 있는 전역 변수 action_items가 없습니다."
    assert actionItems is not None, "actionItems가 제공되지 않았습니다."
    
    action_items_to_tasks_prompt = ChatPromptTemplate.from_template(
    """
    당신은 액션 아이템을 태스크로 변환해 주는 AI 비서입니다. 당신의 주요 언어는 한국어입니다.
    당신의 업무는 작업 내용, 작업 담당자, 작업 마감기한 정보가 담겨 있는 {previous_action_items}와 사용자가 선택한 작업 내용인 {actionItems}를 바탕으로
    프로젝트에 추가될 task의 title, description, assignee, endDate, epicId 정보를 구성하는 것입니다.
    다음의 과정을 따라서 task의 내용을 구성하고, 반드시 {actionItems}에 존재하는 모든 task를 처리하도록 하세요.
    
    1. 우선 {actionItems}에 있는 task 중에 {previous_action_items}에 존재하는 task가 있는지 string을 비교해서 확인하세요.
    이릉이 완전히 동일하지 않더라도 비슷한 내용을 가지고 있다면 동일한 task로 간주해서 처리해 주세요. 이때 cosine similarity를 기준으로 0.9 이상만 동일한 task로 처리합니다.
    이렇게 찾은 동일한 task를 detected_task, 그렇지 않은 task를 undetected_task로 명명하겠습니다.
    2. detected_task와 undetected_task 모두 {actionItems}의 string을 title로 설정하세요. detected와 undetected의 이름이 다르더라도 {actionItems}가 기준입니다.
    3. detected_task와 undetected_task의 title을 가지고 description을 생성하세요.
    4. detected_task의 assignee가 null이 아니라면 {project_members}를 참고해서 이미 선언되어 있는 assignee 정보게 맞게 새롭게 project member 안에서 assignee를 부여해 주세요.
    assignee는 원칙적으로 한 명이어야 하고, 만약 선언되어 있는 assignee가 한 명의 이름이 아닌 "프로젝트 멤버 아무나" 혹은 "누구나 상관없음"과 같이 불특정 다수이 경우, {project_members} 중에 랜덤으로 한 명을 선택하세요.
    5. detected_task의 endDate가 null이 아니라면 그대로 task의 endDate로 설정하세요. 만약 null이라면 그대로 null을 반환하세요.
    6. 1-5번의 과정이 완료되었다면, {epics_str}에서 작업 내용과 가장 관련성이 높아 보이는 에픽의 이름을 찾아서 해당 에픽의 epicId를 epicId 필드값으로 반환해 주세요. 
    제공되는 {epics_str}은 '- 에픽 이름: 에픽 ID' 형식이며 epicId 필드값이 반드시 명시되도록 하세요.
    
    결과를 다음과 같은 형식으로 반환해 주세요:
    {{
        "tasks": [
            {{
                "title": "string",
                "description": "string",
                "assignee": "string",
                "endDate": "string" | null,
                "epicId": "string"
            }},
            ...
        ]
    }}
    """)
    epic_collection = await get_epic_collection()
    epics = await epic_collection.find({"projectId": project_id}).to_list(length=None)
    epics_str = "\n".join([f"- {epic['title']}: {epic['_id']}" for epic in epics])
    print(f"정리된 epics_str: {epics_str}")
    
    project_members = await get_project_members(project_id)
    
    messages = action_items_to_tasks_prompt.format(
        previous_action_items=redis_action_items,
        new_action_items=actionItems,
        project_members=project_members,
        epics=epics_str
    )
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,
    )
    response = await llm.ainvoke(messages)
    try:
        content = response.content
        try:
            gpt_result = extract_json_from_gpt_response(content)
        except Exception as e:
            logger.error(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True)
            raise Exception(f"GPT util 사용 중 오류 발생: {str(e)}", exc_info=True) from e
    except Exception as e:
        logger.error(f"GPT API 처리 중 오류 발생: {e}", exc_info=True)
        raise Exception(f"GPT API 처리 중 오류 발생: {str(e)}", exc_info=True) from e
    
    response = gpt_result["tasks"]
    print(f"태스크 결과: {response}")
    
    return response


### ============================== 테스트 코드 ============================== ###
async def test_meeintg_analysis():
    with open('meeting_sample.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 테스트용 project_id 설정
    project_id = "815cf1fa-2c17-44e5-bd0c-4b93832f67ee"
    
    # 액션 아이템 생성 테스트
    logger.info("=== 액션 아이템 생성 테스트 ===")
    action_items = await create_action_items_finetuned(content)
    print(f"생성된 액션 아이템: {action_items}")
    
    # 회의 요약 생성 테스트
    logger.info("\n=== 회의 요약 생성 테스트 ===")
    title = "MVP 기능 범위 및 개발 일정 논의"
    summary = await create_summary(title, content, project_id)
    print("\n생성된 회의 요약:")
    print(summary)
    
if __name__ == "__main__":
    #print(model_for_ner.config.id2label)
    import asyncio
    asyncio.run(test_meeintg_analysis())