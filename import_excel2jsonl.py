import pandas as pd
import json

def excel2json(excel_file:str,sheet_name:str="0",question_col:str='',answer_col:str='',output_file=''):
    df = pd.read_excel(excel_file, sheet_name=sheet_name,usecols=[question_col,answer_col])
    # 如果需要将 JSON 数据保存到文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 将每一行转换为 JSON 对象
            data = {"conversations":[{"from":"human","value":row[question_col]},{"from":"gpt","value":row[answer_col]}]}
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + '\n')


excel2json(excel_file = '/home/admin/tmp/育宠师知识内容.xlsx',
           question_col='标题',
           sheet_name='育宠师知识内容',
           answer_col='内容',
           output_file = '/home/admin/python_projects/MedicalGPT/data/finetune/育宠师知识内容.jsonl')




