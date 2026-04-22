'''
Description: 
Author: anyiran
Date: 2024-10-14 15:32:07
LastEditors: anyiran
LastEditTime: 2024-11-19 13:24:38
'''
from collections import defaultdict


def compute(session_item, session_bundle, predictions):
    session_precision = 0
    session_recall = 0
    coverage_item = 0
    all_hitted_bundle = 0
    for test_id, pred in predictions.items():
        if len(pred) == 0:
            continue
        all_items =  session_item[test_id].split(',')
        all_bundle = session_bundle[test_id]
        hitted_bundle = 0
        for bid, content in pred.items():
            # print(content)
            # print(all_items)
            try:
                reidx_items = set([all_items[int(i[-1])-1] for i in content])
            except Exception as e:
                print(e)
                print(test_id)
            # print(reidx_items)
            for bundle in all_bundle:
                bundle_list = set(bundle[-1].split(','))
                if reidx_items <= bundle_list: 
                    hitted_bundle += 1
                    union_items = len(bundle_list & reidx_items)
                    coverage_item += union_items / len(bundle_list)
                    all_hitted_bundle += 1
                    break
        session_precision += hitted_bundle / len(pred)
        session_recall += hitted_bundle / len(all_bundle)
    
    session_precision /= len(predictions)
    session_recall /= len(predictions)
    coverage = coverage_item / all_hitted_bundle

    return session_precision, session_recall, coverage

def findErrors (session_id, generated_bundles, session_bundles, session_items):
    # feedback_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:str} # 5: unexcepted error
    feedback_dict = defaultdict(list)
    groundtruth_bundles = session_bundles[session_id]
    all_items_session = session_items[session_id].split(',')
    for bid, items in generated_bundles.items():  
        if type(items) == str:
            items = [items]
        # find the neareast groundtruth
        all_similarity = []
        try:
            idx_items = set([all_items_session[int(item[-1])-1] for item in items])
        except Exception as e:
            # print(e)
            # print(session_id)
            feedback_dict[5].append(bid)
            continue
        for bundle in groundtruth_bundles:
            sim_score = len(idx_items&set(bundle[-1].split(','))) / len(idx_items|set(bundle[-1].split(',')))
            all_similarity.append(sim_score)
        index, max_score = max(enumerate(all_similarity), key=lambda pair: pair[1])
        if max_score == 0:
            feedback_dict[1].append(bid)
        elif max_score == 1:
            feedback_dict[0].append(bid)
        else:
            GT_bundle = set(groundtruth_bundles[index][-1].split(','))
            if idx_items <= GT_bundle:
                if len(idx_items) == 1:
                    feedback_dict[4].append(bid)
                else:
                    feedback_dict[3].append(bid)
            else:
                feedback_dict[2].append(bid)
    return feedback_dict

# def compute_user(user_item, user_bundle, predictions):
#     session_precision = 0
#     session_recall = 0
#     coverage_item = 0
#     all_hitted_bundle = 0
#     for user_id, pred in predictions.items():
#         if len(pred) == 0:
#             continue
#         all_items =  user_item[user_id].split(',')
#         all_bundle = user_bundle[user_id]
#         hitted_bundle = 0
#         for bid, content in pred.items():
#             # print(content)
#             # print(all_items)
#             try:
#                 reidx_items = set([all_items[int(i[-1])-1] for i in content])
#             except Exception as e:
#                 print(e)
#                 print(user_id)
#             # print(reidx_items)
#             for bundle in all_bundle:
#                 bundle_list = set(bundle[-1].split(','))
#                 if reidx_items <= bundle_list: 
#                     hitted_bundle += 1
                    
#                     break
#         session_precision += hitted_bundle / len(pred)
#         session_recall += hitted_bundle / len(all_bundle)
    
#     session_precision /= len(predictions)
#     session_recall /= len(predictions)


#     return session_precision, session_recall


def compute_user(user_items, user_bundles, format_res):
    """
    计算用户推荐结果的精确率 (Precision) 和召回率 (Recall)
    
    Args:
        user_items (dict): 用户及其实际拥有商品的字典
        user_bundles (dict): 用户及其实际拥有捆绑包的字典
        format_res (dict): 模型生成的用户推荐结果
    
    Returns:
        session_precision (float): 精确率
        session_recall (float): 召回率
    """
    total_correct = 0  # 正确推荐的商品总数
    total_recommended = 0  # 推荐的商品总数
    total_relevant = 0  # 实际相关的商品总数

    for user_id, recommended_bundles in format_res.items():
        # 获取用户的实际商品和捆绑包
        actual_items = set(user_items.get(user_id, []))  # 用户实际拥有的商品
        actual_bundles = user_bundles.get(user_id, [])  # 用户实际拥有的捆绑包
        actual_bundles_flat = set(item for bundle in actual_bundles for item in bundle)  # 扁平化的所有实际商品集合

        # 推荐结果中的商品集合
        recommended_items = set(item for bundle in recommended_bundles for item in bundle)

        # 正确推荐的商品
        correct_recommendations = recommended_items.intersection(actual_bundles_flat)

        # 更新统计值
        total_correct += len(correct_recommendations)
        total_recommended += len(recommended_items)
        total_relevant += len(actual_bundles_flat)

    # 计算 Precision 和 Recall
    session_precision = total_correct / total_recommended if total_recommended > 0 else 0.0
    session_recall = total_correct / total_relevant if total_relevant > 0 else 0.0

    return session_precision, session_recall


def findErrors_user (session_id, generated_bundles, session_bundles, session_items):
    # feedback_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:str} # 5: unexcepted error
    feedback_dict = defaultdict(list)
    groundtruth_bundles = session_bundles[session_id]
    all_items_session = session_items[session_id].split(',')
    for bid, items in generated_bundles.items():  
        if type(items) == str:
            items = [items]
        # find the neareast groundtruth
        all_similarity = []
        try:
            idx_items = set([all_items_session[int(item[-1])-1] for item in items])
        except Exception as e:
            # print(e)
            # print(session_id)
            feedback_dict[5].append(bid)
            continue
        for bundle in groundtruth_bundles:
            sim_score = len(idx_items&set(bundle[-1].split(','))) / len(idx_items|set(bundle[-1].split(',')))
            all_similarity.append(sim_score)
        index, max_score = max(enumerate(all_similarity), key=lambda pair: pair[1])
        if max_score == 0:
            feedback_dict[1].append(bid)
        elif max_score == 1:
            feedback_dict[0].append(bid)
        else:
            GT_bundle = set(groundtruth_bundles[index][-1].split(','))
            if idx_items <= GT_bundle:
                if len(idx_items) == 1:
                    feedback_dict[4].append(bid)
                else:
                    feedback_dict[3].append(bid)
            else:
                feedback_dict[2].append(bid)
    return feedback_dict