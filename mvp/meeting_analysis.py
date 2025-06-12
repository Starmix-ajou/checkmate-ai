import logging
import os
from collections import defaultdict
from typing import List

import torch
from dotenv import load_dotenv
from gpt_utils import extract_json_from_gpt_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mongodb_setting import (get_epic_collection, get_project_collection,
                             get_user_collection)
from openai import AsyncOpenAI
from project_member_utils import get_project_members
from transformers import AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

#HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
#login(token=HUGGINGFACE_API_KEY)


'''
Token은 BIO 형식 기반으로 처리. BIO 형식은 각 토큰에 대해 태그 부여 방식을 정의한 것.

B: Begin, I: Inside, O: Outside
B-PER: 시작 토큰, I-PER: 중간 토큰, O: 그 외 토큰
'''


original_model_name = "monologg/koelectra-base-v3-naver-ner"
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

tokenizer = AutoTokenizer.from_pretrained(original_model_name)
model_for_ner = AutoModelForTokenClassification.from_pretrained(original_model_name)

### ==================== 회의 액션 아이템 생성 - 파인튜닝 모델 사용 ==================== ###
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
    logger.info(f"최종 처리된 action_items: {action_items}")
    
    return action_items
    
### ============================== API 정의 ============================== ###
### ================ Summary & Action Items Extraction ================== ###
async def create_summary(title: str, content: str, project_id: str):
    '''
    title: 사용자가 제목으로 회의록을 대표하는 내용을 입력한다고 가정 -> 요약의 첫 번째 뼈대로 사용
    content: Markdown 형태로 문서가 제공됨
    '''
    logger.info(f"🔍 회의 요약 생성 시작")
    meeting_summary_prompt = ChatPromptTemplate.from_template("""
    당신은 회의록에서 중요한 대화 내용을 정리해 주는 AI 비서입니다. 당신의 주요 언어는 한국어입니다. 정리한 내용은 Markdown 형식으로 반환해 주세요.
    당신의 업무는 회의 제목인 {title}을 바탕으로 회의록 {content}를 분석하여 중요한 대화 내용을 정리하는 것입니다.
    {title}은 회의의 제목으로서 회의록에서 논의되는 내용을 대표하는 것으로 간주합니다. 
    회의록의 내용을 분석할 때 {title}을 적극적으로 참조하고, 요약본의 맨 앞에 Heading 1 레벨로 {title}을 불렛 포인트 없이 넣으세요.
    
    {content}에 포함된 token의 수가 3000개 이상을 넘어가면 회의 안건, 안건 논의 결과, 다음 회의 안건, 중요 피드백 및 의견 정리 등의 목차를 구성하여 목차별로 체계적으로 정리하세요.
    3000개 미만의 짧은 회의록의 경우에는 내용을 500자 이내로 최대한 압축해서 정리하세요. 이 때 목차를 구성하지 말고 내용을 최대한 압축해서 정리하세요.
    
    {content}에 포함된 Heading 레벨 표시 문자인 "#", "**" 등의 특수 문자는 요약을 구성하는 과정에만 참고하고 요약 결과에는 포함하지 마세요.
    요약 결과는 내용을 보기 쉽게 불렛 포인트와 함께 문장으로 정리하세요.
    
    반드시 다음의 JSON 형식으로만 응답해 주세요. 다른 형식의 응답은 허용되지 않습니다. 다시 말하지만 반드시 JSON 형식으로만 응답해 주세요.
    또한 반드시 summary를 Markdown 형식으로 작성하세요:
    {{
        "summary": "여기에 요약 내용을 Markdown 형식으로 작성"
    }}
    """)
    
    project_members = await get_project_members(project_id)
    
    messages = meeting_summary_prompt.format(
        title=title,
        content=content,
        project_members=project_members)
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.8,
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
    logger.info(f"회의 요약 결과: {summary}")
    
    return summary

async def create_action_items_gpt(content: str):
    logger.info(f"🔍 회의 액션 아이템 생성 시작")
    action_items_prompt = ChatPromptTemplate.from_template("""
    당신은 회의록으로부터 액션 아이템을 추출해서 정리하는 AI 비서입니다. 당신의 주요 언어는 한국어입니다.
    회의록 {content}를 분석하여 다음 세 가지 요소를 포함한 액션 아이템을 추출해 주세요:
    1. 액션 아이템 내용 (description)
    2. 담당자 (assignee)
    3. 마감 기한 (endDate)
    2번과 3번은 회의록에 정보가 없을 경우 null로 지정하세요.
    액션 아이템의 내용은 "~하기"로 명사형 어미를 사용해서 작성해야 합니다. 이를 위해 description을 한 번 더 정리하는 과정을 거치세요.
    
    결과를 다음과 같은 JSON 형식으로 반환해 주세요. 다른 형식의 응답은 허용되지 않습니다. 다시 말하지만 반드시 JSON 형식으로만 응답해 주세요.
    {{
        "actionItems": [
            {{
                "description": "string",
                "assignee": "string" | null,
                "endDate": "string" | null
            }},
            ...
        ]
    }}
    """)
    messages = action_items_prompt.format(content=content)
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.5,
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
    logger.info(f"생성된 액션 아이템: {action_items}")
    
    return action_items


async def convert_action_items_to_tasks(action_items: List[str], project_id: str):
    assert action_items is not None, "action_items가 제공되지 않았습니다."
    
    action_items_to_tasks_prompt = ChatPromptTemplate.from_template(
    """
    당신은 주어진 액션 아이템의 세부 내용을 정리해서 task로 변환하는 AI 비서입니다. 당신의 주요 언어는 한국어입니다.
    당신의 업무는 작업 내용, 작업 담당자, 작업 마감기한 정보가 담겨 있는 {action_items}로부터 title, description, assignee, endDate, epicId의 정보를 완성하는 것입니다.
    반드시 다음의 과정을 따라서 {action_items}에 존재하는 item을 하나씩 처리하고, 모든 item이 처리되도록 하세요.
    1. {action_items}에서 key값으로 description, assignee, endDate가 존재하는 다음 item을 선택해서 assignee와 endDate가 null인지 확인하세요.
    2. assingee가 null인 경우 null을 값으로 그대로 반환하고, null이 아닌 경우 assignee가 {project_members}에 속한 구성원인지 확인하세요.
    assignee가 멤버의 이름이 아닌 position의 이름일 수 있으므로 {project_members}로부터 멤버의 이름과 position 정보를 모두 확인하고, position이 assginee에 적혀 있는 경우 구성원의 이름으로 대체하여 반환하세요.
    멤버의 이름과 position 정보가 모두 일치하지 않는 경우에만 assignee 값으로 null을 반환합니다.
    3. endDate는 endDate가 null인 경우 null을 값으로 그대로 반환하고, null이 아닌 경우 endDate가 오늘 날짜 이후인지 확인하세요. 만약 오늘 날짜 이후가 아닌 경우 endDate 값으로 null을 반환합니다.
    endDate는 datetime 형식을 지닌 string으로 반환하세요.
    4. description을 10글자 이내로 요약하여 title을 구성하세요.
    5. {epics}에는 프로젝트에 속한 모든 epic들의 title, description, id 정보가 다음과 같은 형식으로 정리되어 있습니다: "내용: (description) --- id: (ObjectId)"
    epic별 description을 바탕으로 현재 item의 내용과 가장 유사한 epic을 {epics} 목록 안에서 선택하세요. 이 때 '유사하다'의 정의는 epic의 description과 item의 description 간의 cosine similarity가 0.95 이상임을 의미합니다.
    만약 유사한 epic이 선택되지 않은 경우에는 cosine similarity의 threshold를 0.95에서 0.90으로 낮춰서 다시 유사한 epic을 선택하세요.
    이 때도 유사한 epic이 선택되지 않으면 threshold를 한 번 더 0.90에서 0.80으로 조정합니다.
    그럼에도 선택되지 않는다면 null을 반환하세요.
    6. 5번에서 선택한 epic의 id를 epicId로 반환하세요. 이때 직접 epicId를 생성하는 게 아니라 반드시 {epics}에 저장되어 있는 id 값을 그대로 반환해야 합니다. 한 번 더 강조합니다. 절대 epicId를 임의로 생성하지 말고 있는 정보를 그대로 입력하세요.
    
    결과를 다음과 같은 JSON 형식으로 반환해 주세요. 다른 형식의 응답은 허용되지 않습니다. 다시 말하지만 반드시 JSON 형식으로만 응답해 주세요.
    {{
        "actionItems": [
            {{
                "title": "string",
                "description": "string",
                "assigneeId": "string" | null,
                "endDate": "string" | null,
                "epicId": "string"
            }},
            ...
        ]
    }}
    """)
    epic_collection = await get_epic_collection()
    epics = await epic_collection.find({"projectId": project_id}).to_list(length=None)
    epics_content = "\n".join([f"epic_description: {epic['description']} --- epic_id: ({epic['_id']})" for epic in epics])  # epic들의 title, description, id 정보를 문자열로 정리
    #logger.info(f"정리된 epics_content: {epics_content}")
    
    project_members = await get_project_members(project_id)
    
    messages = action_items_to_tasks_prompt.format(
        action_items=action_items,
        project_members=project_members,
        epics=epics_content
    )
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.2,
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
    
    response = gpt_result["actionItems"]
    logger.info(f"actionItems 구성 결과: {response}")
    
    # assignee 이름을 대응되는 id로 변경
    name_to_id = {}
    user_collection = await get_user_collection()
    project_collection = await get_project_collection()
    project_data = await project_collection.find_one({"_id": project_id})   # DBRef에서 직접 ID 매핑 생성
    logger.info("🔍 프로젝트 멤버 name:id mapping 시작")
    for member_ref in project_data["members"]:
        try:
            user_id = member_ref.id
            user_info = await user_collection.find_one({"_id": user_id})
            if user_info is None:
                logger.warning(f"⚠️ 사용자 정보를 찾을 수 없습니다: {user_id}")
                continue
            
            name = user_info.get("name")
            if name is None:
                logger.warning(f"⚠️ 사용자 이름이 없습니다: {user_id}")
                continue
            
            # ObjectId를 문자열로 변환
            name_to_id[name] = str(user_id)
            logger.info(f"✅ 사용자 매핑 성공 - 이름: {name}, ID: {str(user_id)}")
        except Exception as e:
            logger.error(f"❌ 사용자 정보 처리 중 오류 발생: {str(e)}", exc_info=True)
            continue
    
    assert name_to_id is not None, "name_to_id 매핑 정보가 구성되지 않았습니다."    # mapping 여부 검증
    
    for item in response:
        try:
            # 담당자를 이름:id mapping
            if item["assigneeId"] is None:
                logger.info(f"📌 {item['description']}의 담당자가 null입니다.")
                #continue
            elif item["assigneeId"] in name_to_id:
                logger.info(f"✅ {item['title']}의 담당자인 {item['assigneeId']}가 매핑된 name_to_id에 존재합니다.")
                item["assigneeId"] = name_to_id[item["assigneeId"]]
            else:
                logger.info(f"⚠️ {item['title']}의 담당자가 {item['assigneeId']}로 존재하지만 name_to_id에 매핑된 정보가 없습니다.")
                item["assigneeId"] = None
        except Exception as e:
            logger.error(f"name_to_id 매핑 처리 중 오류 발생: {str(e)}", exc_info=True)

        # epic이 올바르게 연결되었는지 확인
        try:
            if item["epicId"] is not None:
                logger.info(f"✅ {item['title']}에 매핑된 epicId가 존재합니다. epicId: {item['epicId']}")
                try:
                    selected_epic = await epic_collection.find_one({"_id": item["epicId"]})
                    logger.info(f"🔍 epicId를 사용해서 epic collection으로부터 조회된 epic 제목: {selected_epic['title']}")
                except Exception as e:
                    logger.warning(f"⚠️ 액션 아이템에 할당된 epicId가 존재하지만 실제 epic collection에서 조회되지 않습니다. 오류 내용: {str(e)}", exc_info=True)
                    item["epicId"] = None
            else:
                logger.info(f"🔍 {item['title']}에 매핑된 epic이 없습니다.")
        except Exception as e:
            logger.error(f"epicId 매핑 처리 중 오류 발생: {str(e)}", exc_info=True)
            
        # endDate가 null을 반환하는 경우, 'null'이 아닌 null을 제대로 반환하는지 확인
        try:
            if item["endDate"] == 'null':
                logger.warning(f"⚠️ {item['title']}의 endDate가 string 'null'로 되어 있습니다.")
                item["endDate"] = None
        except Exception as e:
            logger.error(f"endDate 매핑 처리 중 오류 발생: {str(e)}", exc_info=True)

    logger.info(f"🔍 다음이 API의 response로 반환됩니다: {response}")
    return response

### ============================== 메인 routing 함수 ============================== ###
async def analyze_meeting_document(title: str, content: str, project_id: str):
    '''
    # md 파일에 대한 요약 생성
    - 결과를 md 파일로 반환해야 함
    - 원본에 있는 Heading 레벨을 요약본에서도 유지해야 함
    - 회의 제목을 바탕으로 요약본이 동일한 방향성을 띄는지 정성적으로 평가
    
    '''   
    summary = await create_summary(title, content, project_id)
    logger.info(f"✅ 생성된 회의 요약: {summary}")
    
    '''
    # 액션 아이템 생성
    - 요약한 결과에 액션 아이템을 포함해서 분석할지 아니면 원본에서부터 시작할지 결정해야 함 (일단 후자로 진행)
    - 액션 아이템의 내용으로 (task, assingee, endDate) 쌍의 집합을 반환받아야 함
    - 반환된 action_items를 convert_action_items_to_tasks 함수에 인수로 전달
    '''
    action_items = await create_action_items_gpt(content)
    logger.info(f"✅ 생성된 액션 아이템: {action_items}")
    
    '''
    # action_items를 task로 변환
    - action items에는 task 기준에서 description, assignee, endDate 정보가 담겨 있음
    - 이를 바탕으로 title, epicId를 부여하고 assingee 이름을 대응되는 id로 변경해야 함
    - 단, assignee, endDate가 null일 수 있는데 이 경우에는 일단 null로 모두 반환 -> 이후에 추가 처리 필요
    '''
    actionItems = await convert_action_items_to_tasks(action_items, project_id)
    logger.info(f"✅ task로 변환된 액션 아이템: {actionItems}")
    
    response = {
        "summary": summary,
        "actionItems": actionItems,
    }
    logger.info(f"구성된 response: {response}")
    return response


### ============================== 테스트 코드 ============================== ###
async def test_meeintg_analysis():
    with open('meeting_sample.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 테스트용 project_id 설정
    project_id = "b5728b16-6610-4762-b178-bb71f56a6616"
    
    title = "꼼꼼한 회의록"
    summary = await create_summary(title, content, project_id)
    print(f"생성된 회의 요약: {summary}")
    
    # 회의 요약 생성 테스트
    # load_dotenv()
    # print("\n=== 회의 요약 생성 테스트 - 꼼꼼한 회의록 버전 ===")
    # title = "꼼꼼한 회의록"
    # print("\n 원본: \n")
    # with open('meeting_sample_strict.md', 'r', encoding='utf-8') as f:
    #     content = f.read()
    # print(content)
    # summary = await create_summary(title, content, project_id)
    # print(f"생성된 회의 요약: {summary}")
    
    # # 액션 아이템 생성 테스트
    # print("=== 액션 아이템 생성 테스트 - 꼼꼼한 회의록 버전 ===")
    # action_items = await create_action_items_gpt(content)
    # print(f"생성된 액션 아이템: {action_items}")
    
    
    # print("\n=== 회의 요약 생성 테스트 - 느슨한 회의록 ===")
    # title = "느슨한 회의록"
    # print("\n 원본: \n")
    # with open('meeting_sample_rough.md', 'r', encoding='utf-8') as f:
    #     content = f.read()
    # print(content)
    # summary = await create_summary(title, content, project_id)
    # print(f"생성된 회의 요약: {summary}")
    
    # # 액션 아이템 생성 테스트
    # print("=== 액션 아이템 생성 테스트 - 느슨한 회의록 버전 ===")
    # action_items = await create_action_items_gpt(content)
    # print(f"생성된 액션 아이템: {action_items}")


if __name__ == "__main__":
    #print(model_for_ner.config.id2label)
    import asyncio
    asyncio.run(test_meeintg_analysis())