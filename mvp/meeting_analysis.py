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
from mongodb_setting import (get_epic_collection, get_project_collection,
                             get_user_collection)
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


### ============================== Summary & Action Items Extraction ============================== ###
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
    project_members_str = "\n".join([f"- {member}" for member in project_members])
    
    messages = meeting_summary_prompt.format(
        title=title, 
        content=content,
        project_members=project_members_str)
    
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
        await save_to_redis(f"action_items:{project_id}", action_items)
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
        action_items = await load_from_redis(f"action_items:{project_id}")
    except Exception as e:
        logger.error(f"action_items 로드 중 오류 발생: {str(e)}", exc_info=True)
        raise Exception(f"action_items 로드 중 오류 발생: {str(e)}", exc_info=True) from e
    
    assert action_items is not None, "정의되어 있는 전역 변수 action_items가 없습니다."
    
    try:
        # 가지고 있는 action_items에서 actionItems에 존재하지 않는 task를 제거
        tasks_to_remove = [item for item in action_items if item["task"] not in actionItems]
        print(f"제거할 task: {tasks_to_remove}")
        for task in tasks_to_remove:
            action_items.remove(task)
        print(f"최종 남은 task: {action_items}")
        assert len(action_items) == len(actionItems), "action_items와 actionItems의 개수가 다릅니다."
    except Exception as e:
        logger.error(f"action_items 업데이트 중 오류 발생: {str(e)}", exc_info=True)
        raise Exception(f"action_items 업데이트 중 오류 발생: {str(e)}", exc_info=True) from e
    
    action_items_to_tasks_prompt = ChatPromptTemplate.from_template(
    """
    당신은 액션 아이템을 태스크로 변환해 주는 AI 비서입니다. 당신의 주요 언어는 한국어입니다.
    당신의 업무는 작업 내용, 작업 담당자, 작업 마감기한 정보가 담겨 있는 {action_items}를 바탕으로 tasks의 title, description, assignee, endDate, epicId 정보를 구성해 주는 것입니다.
    다음의 과정을 따라 각 task의 내용을 구성해 주세요. 반드시 {action_items}에 존재하는 모든 task를 처리해야 합니다.
    1. "task" 필드값을 바탕으로 작업명(title)을 설정해 주세요.
    2. "task" 필드값을 바탕으로 작업 설명(description)을 설정해 주세요.
    3. "assignee" 필드값을 바탕으로 작업 담당자(assignee)를 설정해 주세요. 작업 담당자 정보가 없다면 description을 바탕으로 {project_memebers} 안에서 적절한 담당자를 골라서 설정해 주세요.
    4. "enddate" 필드값을 바탕으로 작업 마감기한(endDate)를 설정해 주세요. 만약 {action_items}에 "enddate" 필드값이 없다면 null을 반환하세요.
    5. 1-4번의 과정이 완료되었다면, {epics_str}에서 작업 내용과 가장 관련성이 높아 보이는 에픽의 이름을 찾아서 해당 에픽의 epicId를 epicId 필드값으로 반환해 주세요. 반드시 epicId 필드값이 명시되어야 하며, 
    제공되는 {epics_str}은 '- 에픽 이름: 에픽 ID' 형식으로 제공됩니다.
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
    
    epics = await get_epic_collection().find_many({"projectId": project_id})
    epics_str = "\n".join([f"- {epic['title']}: {epic['_id']}" for epic in epics])
    project_members = await get_project_members(project_id)
    
    messages = action_items_to_tasks_prompt.format(
        action_items=action_items,
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
    with open('mvp/meeting_sample.md', 'r', encoding='utf-8') as f:
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