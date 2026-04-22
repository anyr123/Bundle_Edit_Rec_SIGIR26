'''
Description: 
Author: anyiran
Date: 2024-10-14 15:32:07
LastEditors: anyiran
LastEditTime: 2024-11-02 17:16:18
'''
import re
import ast

def output_parser(response_str, type='bundle'):
    state_code = 0
    response_str = response_str.replace('\n', '')
    if type == 'bundle':
        if '[' in response_str:
            if not response_str.startswith('{'):
                match_str = re.search(r'{.*}', response_str, re.DOTALL)
                if match_str is None:
                    # 'Since there are no bundles after adjustment, intents cannot be generated.'
                    
        
                    match_str="{}"
                else:
                    match_str = match_str.group().replace('\n', '')
                response_str = match_str
            
            response_str = response_str.replace("'",'"')
            # 用双引号替换不带 's ' 的单引号
            response_str = re.sub(r'"s(\b)', r"'s\1", response_str)
            response_str = re.sub(r's"(\s\b)', r"s'\1", response_str)
            
            pattern = r'"([^"]+)": (?:\[(.*?)\]|"(.*?)")'
            if ":-" in response_str:
                pattern=r'([^-:]+): (?:\[(.*?)\]|"(.*?)")'
            result = {}
            # 匹配每个 bundle 中的内容
            matches = re.findall(pattern, response_str)
            print(matches)

            # 遍历每个匹配项，提取 product 编号
            for bundle, products1, products2 in matches:
                # 根据哪个匹配成功来选择产品列表
                products = products1 or products2
                product_list = re.findall(r'product\d+', products)
                if bundle not in result:
                    result[bundle] = []
                # Add the product to the list for the current bundle
                for p in product_list:
                    result[bundle].append(p)    
            response_dict=result      
            # response_dict = ast.literal_eval(response_str)
            state_code = 200
        else:
            state_code = 404
            response_dict = {}
        #经过匹配仍然为空字典的话，404
        # if response_dict=={}:
        #     state_code=404
    
            
    elif type == 'intent':
        # if response_str
        if response_str is None:
            print("没有得到response")
        if not response_str.startswith('{'):
            match_str = re.search(r'{.*}', response_str, re.DOTALL)
            if match_str is None:
                # 'Since there are no bundles after adjustment, intents cannot be generated.'
                match_str="{}"
            else:
                match_str = match_str.group().replace('\n', '')
            response_str = match_str
        
        response_str = response_str.replace("'",'"')
        # 用双引号替换不带 's ' 的单引号
        response_str = re.sub(r'"s(\b)', r"'s\1", response_str)
        response_str = re.sub(r's"(\s\b)', r"s'\1", response_str)

        # 定义正则表达式模式来匹配 'bundle X' 以及其中的 product 编号
        pattern = r'"([^"]+)": "([^"]+)"'

        # 匹配每个 bundle 中的内容
        matches = re.findall(pattern, response_str)
        print(matches)
        # 初始化结果字典
        result = {}
        # 遍历每个匹配项，提取 product 编号
        for bundle, intent in matches:    
            result[bundle] = intent  # 去掉多余的空格
        response_dict=result
        
        state_code = 200

    return {'state_code': state_code, 'output': response_dict}

def process_results(bundle_res):
    invalid_id = []
    for testid, bundles in bundle_res.items():
        c = 0
        for b,items in bundles.items():
            if len(items)==1:
                c+=1
        # print(c, len(bundles))
        if c==len(bundles):
            # print(test_id)
            invalid_id.append(testid)
    print(invalid_id)

    remove_invalid_res = {}
    for test_id, bundles in bundle_res.items():
        if test_id in invalid_id:
            continue
        format_bundles = {}
        for bid, items in bundles.items():
            if len(items)>1:
                format_bundles[bid] = items
        remove_invalid_res[test_id] = format_bundles

    return remove_invalid_res