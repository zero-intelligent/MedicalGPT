import pandas as pd
import json
from model_predict import predict

def excel2json_qa_conversation(
    excel_file:str,
    sheet_name:str|int=0,
    question_col:str='Question',
    answer_col:str='Answer',
    output_file=''
):
    df = pd.read_excel(excel_file, sheet_name=sheet_name,usecols=[question_col,answer_col])
    # 如果需要将 JSON 数据保存到文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 将每一行转换为 JSON 对象
            data = {"conversations":[{"from":"human","value":row[question_col]},{"from":"gpt","value":row[answer_col]}]}
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + '\n')


def excel2json_multi_loop_conversation(
    excel_file:str,
    sheet_name:str|int=0,
    session_id_col:str='提问id',
    question_col:str='宠物类型,宠物年龄,问题描述',
    ack_seq_col:str='回复id',
    answer_col:str='对话内容',
    role_col:str = "对话类型",
    doctor_role:str = "医生回复",
    patient_role:str = "用户追问",
    output_file=''
):
    usecols = [session_id_col,ack_seq_col,answer_col,role_col] + question_col.split(',')
    df = pd.read_excel(excel_file, sheet_name=sheet_name,usecols=usecols)
    
    def process_group(group_items):
        # 排序
        sorted_group_items = group_items.sort_values(by=ack_seq_col, ascending=True)
        
        #组合问题
        question = ','.join([f"{q}:{sorted_group_items[q].iloc[0]}" for q in question_col.split(',')])
        
        #生成会话
        dialogs = [{
            "from":"human",
            "value":question
        }]
        current_role = 'gpt'
        for _, row in sorted_group_items.iterrows():
            if type(row[answer_col]) != str:
                print('answer_col type mismtch')
                continue
                    
            role_swap = (current_role == 'gpt' and row[role_col] == doctor_role \
                or current_role == 'human' and row[role_col] == patient_role)
            
            # 交换角色
            if role_swap:
                dialogs.append({
                    "from": current_role,
                    "value":row[answer_col]
                })
                current_role = 'gpt' if current_role == 'human' else 'human'
            else:
                dialogs[-1]["value"] += "\n" + row[answer_col]
                
        
        return {"conversations":dialogs}

    grouped_result = df.groupby(session_id_col).apply(process_group).reset_index(drop=True)

    # 如果需要将 JSON 数据保存到文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in grouped_result:
            # 将每一行转换为 JSON 对象
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + '\n')
            

def batch_predict(
    excel_file:str,
    sheet_name:str|int=0,
    question_col:str='question',
    answer_col:str='sft_result',
    output_file=''
):
    def apply_row(question):
        return predict(question)
    
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df[answer_col] = df[question_col].apply(apply_row)
    if not output_file:
        output_file = excel_file
    df.to_excel(output_file, index=False)


# excel2json_multi_loop_conversation(
#             excel_file = '/home/admin/tmp/历史问诊对话数据.xlsx',
#             session_id_col = '提问id',
#             question_col='宠物类型,宠物年龄,问题描述',
#             ack_seq_col='回复id',
#             answer_col='对话内容',
#             role_col = "对话类型",
#             doctor_role = "医生回复",
#             patient_role = "用户追问",
#             output_file='/home/admin/python_projects/MedicalGPT/data/finetune/历史问诊对话数据.jsonl')

# excel2json_qa_conversation(excel_file = '/home/admin/tmp/育宠师知识内容.xlsx',
#            question_col='标题',
#            sheet_name='育宠师知识内容',
#            answer_col='内容',
#            output_file = '/home/admin/python_projects/MedicalGPT/data/finetune/育宠师知识内容.jsonl')


batch_predict(excel_file = '/home/admin/tmp/宠物医疗评估数据.xlsx',
           question_col = '病人',
           answer_col = 'SFT模型回复结果')