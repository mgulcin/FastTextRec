'''
Created on Mar, 2018

@author: mgulcin
'''

import csv
import math
import os

import Common


def prune_users(user_2_list_of_visits_single, trainDataFilePath):
    # select users: who has check-in in both Jan and Feb && whose checkin location in Feb is seen in January
    train_users_2_locations = Common.readTrainData(trainDataFilePath)
    all_jan_users = train_users_2_locations.keys()
    all_jan_locations = train_users_2_locations.values()
    pruned_users_2_list_of_visits_single = {user: user_2_list_of_visits_single[user] for user in user_2_list_of_visits_single.keys() if user in all_jan_users}
    
    return pruned_users_2_list_of_visits_single.keys()
    
def read_true_visits_seq(test_pruned_output_file_path):
    user_2_list_of_visits_seq = {}
    user_2_list_of_visits_single = {}
    with open(test_pruned_output_file_path, "rb") as f:
        reader = csv.reader(f, delimiter='\t', quoting = csv.QUOTE_NONE)
        for row in reader:
            # remove [, ]
            updated_row = [value.replace("[", "") for value in row]
            updated_row = [value.replace("]", "") for value in updated_row]
            # use _ as the separator
            updated_row = [value.replace(", ", "_") for value in updated_row]
            
            user_id = updated_row[0]
            list_of_visits_seq = updated_row[1:]
            if len(list_of_visits_seq) >0:
                user_2_list_of_visits_seq[user_id] = list_of_visits_seq #list(set(list_of_visits_seq))
            
            list_of_visits_single = []
            for visit_seq in list_of_visits_seq:
                items = visit_seq.split('_')
                list_of_visits_single.extend(items)
            if len(list_of_visits_single) >0:
                user_2_list_of_visits_single[user_id]= list_of_visits_single#list(set(list_of_visits_single))
            
            #print user_2_list_of_visits_seq
            #print user_2_list_of_visits_single
            #break
    return user_2_list_of_visits_seq, user_2_list_of_visits_single

def read_rec_seq(rec_seq_output_folder, rec_output_file):
    rec_seq_output_file_path = os.path.join(rec_seq_output_folder, rec_output_file)
    
    user_2_rec_seq = {} ## user1253,['13035', '35315_13133']
    user_2_rec_seq_to_single = {} ## user1253,['13035', '35315', '13133']
    user_2_rec_seq_to_first = {}## user1253,['13035','35315']
    with open(rec_seq_output_file_path, "rb") as f:
        reader = csv.reader(f, delimiter=',')#, quotechar = "'"
        for row in reader:
            # remove ' and space
            updated_row = [value.replace("'", "") for value in row]
            updated_row = [value.replace(" ", "") for value in updated_row]
            # remove [, ]
            updated_row = [value.replace("[", "") for value in updated_row]
            updated_row = [value.replace("]", "") for value in updated_row]
            
            # remove "loc"
            updated_row = [value.replace("loc", "") for value in updated_row]
            
            user_id = updated_row[0]
            rec_seq_output = updated_row[1:]
            user_2_rec_seq[user_id] = rec_seq_output
            
            rec_seq_to_single_output = []
            rec_seq_to_first = []
            for rec_seq in rec_seq_output:
                items = rec_seq.split('_')
                rec_seq_to_single_output.extend(items)
                rec_seq_to_first.append(items[0])
            user_2_rec_seq_to_first[user_id] = rec_seq_to_first
            user_2_rec_seq_to_single[user_id]= rec_seq_to_single_output
            
    return user_2_rec_seq, user_2_rec_seq_to_single, user_2_rec_seq_to_first

def read_rec_single(rec_output_folder, rec_output_file):
    rec_output_file_path = os.path.join(rec_output_folder, rec_output_file)
    
    user_2_rec_single = {} ## user1253,['13035', '35315', '13133']
    with open(rec_output_file_path, "rb") as f:
        reader = csv.reader(f, delimiter=',')#, quotechar = "'"
        for row in reader:
            # remove ' and space
            updated_row = [value.replace("'", "") for value in row]
            updated_row = [value.replace(" ", "") for value in updated_row]
            # remove [, ]
            updated_row = [value.replace("[", "") for value in updated_row]
            updated_row = [value.replace("]", "") for value in updated_row]
            # remove "loc"
            updated_row = [value.replace("loc", "") for value in updated_row]
            
            user_id = updated_row[0]
            rec_output = updated_row[1:]
            user_2_rec_single[user_id] = rec_output
    return user_2_rec_single
    
def count_tp(rec_output_type, y_true_single_user, y_pred_single_user):
    if rec_output_type == "Seq":
        ##E.g. user1253,['13035', '35315_13133']
        return count_tp_seq(y_true_single_user, y_pred_single_user)
    elif rec_output_type == "Seq_Single":
        ##E.g. user1253,['13035', '35315', '13133']
        return count_tp_seq_single(y_true_single_user, y_pred_single_user)
    elif rec_output_type == "Seq_First":
        ##E.g. user1253,['13035', '35315']
        return count_tp_seq_single(y_true_single_user, y_pred_single_user)
    elif rec_output_type == "Single":
        ##E.g. user1253,['13035', '35315', '13133']
        return count_tp_single(y_true_single_user, y_pred_single_user)
    else:
        print("Wrong type @ count_tp")
        exit

def isRelevant(item, y_true):
    if item in y_true:
        return True
    return False
    
def count_tp_seq(y_true_single_user, y_pred_single_user):
    ##E.g. user1253,['13035', '35315_13133']
    tp = 0
    for item in  y_pred_single_user:
        if isRelevant(item, y_true_single_user):
            tp = tp+1
    return tp

def count_tp_seq_single(y_true_single_user, y_pred_single_user):
    ##E.g. user1253,['13035', '35315', '13133']
    tp = 0
    for item in  y_pred_single_user:
        if isRelevant(item, y_true_single_user):
            tp = tp+1
    return tp

def count_tp_single(y_true_single_user, y_pred_single_user):
    ##E.g. user1253,['13035', '35315', '13133']
    tp = 0
    for item in  y_pred_single_user:
        if isRelevant(item, y_true_single_user):
            tp = tp+1
    return tp


# def find_tp_fp(true_rec, pred_rec, user_list):
#     user_2_tp_and_fp = {}
#     for user in user_list:
#         y_true = user_2_list_of_visits_seq[user]
#         y_pred = user_2_rec_seq[user]
#         tp = count_tp(y_true, y_pred)
#         
#         fp = len(y_pred) - tp
#         user_2_tp_and_fp[user] = (tp, fp)
#     return user_2_tp_and_fp

def calculatePrecision(tp, fp):
    return float(tp)/(tp+fp)

def calculateDcg(rec_size, y_true, y_pred):
    # DCG_{p} = rel_1 + \sum_{i=2}^{p} \frac{rel_{i}}{\log_{2}(i+1)} 
    
    rel1 = 0.0
    if isRelevant(y_pred[0], y_true):
        rel1 = 1.0
    total = 0.0;
    for i in xrange(1, min(rec_size, len(y_pred))):
        rel = 0.0
        if isRelevant(y_pred[i], y_true):
            rel = 1.0
        denom = (math.log(i+1)/math.log(2));
        total +=rel/denom
    dcg = rel1 + total
    return dcg

def calculateIdealDcg(rec_size, y_true):
    #sorting documents of a result list by relevance, 
    # producing the maximum possible DCG till position p
    total_count = len(y_true)
    rel1 = 0.0
    if isRelevantIdcg(total_count, 0):
        rel1 = 1.0
    total = 0.0;
    for i in xrange(1, rec_size):
        rel = 0.0
        if isRelevantIdcg(total_count, i):
            rel = 1.0
        denom = (math.log(i+1)/math.log(2));
        total +=rel/denom
    idcg = rel1 + total
    return idcg

def isRelevantIdcg(total_count, i):
    if(i<=total_count):
        return True
    return False    
    
from iteration_utilities import unique_everseen
def evaluate(rec_output_type, user_list, rec_size, user_2_true, user_2_pred):
    user_2_tp_and_fp_and_prec_and_ndcg = {}
    for user in user_list:
        if user == "user37" and rec_output_type=="Single":
            "stop"
        tp = 0.0
        fp = rec_size
        prec = 0.0
        ndcg = 0.0
        
        if user in user_2_true:
            y_true = list(unique_everseen(user_2_true[user]))
            y_pred = list(unique_everseen(user_2_pred[user]))
            tp = count_tp(rec_output_type, y_true, y_pred)
            #fp = len(y_pred) - tp 
            # NOTE Assumed if pred_size< rec_size(k), the missing part is fp
            fp = max(len(y_pred), rec_size) - tp 
            prec = calculatePrecision(tp, fp)
            dcg = calculateDcg(rec_size, y_true, y_pred)
            idcg = calculateIdealDcg(rec_size, y_true)
            ndcg = dcg/idcg;
            
            user_2_tp_and_fp_and_prec_and_ndcg[user] = (tp, fp, prec, ndcg)
        
#     // write precs of eah user
#     printer.printPrecisionEvalResult(folder + outputPrecfileName, controlUserIdList, userPrecMap, userTPMap, userFPMap);
# 
#     // calculate precision for overall results
#     Double precisionTotal = EvaluateCheckin2011DB.precision(truePosTotal, falsePosTotal);
#     String str = "Overall -- tpTotal: " + truePosTotal 
#             + " fpTotal: " + falsePosTotal
#             + " prec: " + precisionTotal;
#     printer.printString(folder + outputPrecfileName, str);
    return user_2_tp_and_fp_and_prec_and_ndcg

def find_overall_precision(user_2_tp_and_fp_and_prec_and_ndcg):   
    tp_sum = 0.0
    fp_sum = 0.0
    for input_tuple in user_2_tp_and_fp_and_prec_and_ndcg.values():
        tp_sum = tp_sum + input_tuple[0]
        fp_sum = fp_sum + input_tuple[1]
    #print(tp_sum, fp_sum)
    return calculatePrecision(tp_sum, fp_sum)

def find_overall_hitrate(user_list, user_2_tp_and_fp_and_prec_and_ndcg):   
    hit_count = 0.0
    total_user_count = 0.0
    for user, input_tuple in user_2_tp_and_fp_and_prec_and_ndcg.iteritems():
        if user in user_list:
            tp =input_tuple[0]
            if(tp>0):
                hit_count= hit_count+1
            total_user_count = total_user_count + 1
    #print(tp_sum, fp_sum)
    return hit_count/total_user_count

def find_overall_ndcg(user_list, user_2_tp_and_fp_and_prec_and_ndcg):   
    ndcg_sum = 0.0    
    total_user_count = 0.0
    for user, input_tuple in user_2_tp_and_fp_and_prec_and_ndcg.iteritems():
        if user in user_list:
            ndcg_sum = ndcg_sum + input_tuple[3]
            total_user_count = total_user_count + 1
    #print("total_user_count: "+str(total_user_count))
    avg_ndcg = ndcg_sum / total_user_count
    return avg_ndcg

def create_rec_output_file_acronomies_list():
    # e.g. rec_output_file = "fasttext_contentbased_rec_SkipGram_VS=200_MaxN=5"
    rec_output_file_acronomies_list = []
    vectorSizes = [10, 50, 100, 150, 200, 250]
    isSGValues = [0, 1]
    defaultVectorSize = 100
    defaultMaxNgram = 5
    for isSG in isSGValues:
        for vectorSize in vectorSizes:
            usedFeaturesAcronym = Common.createUsedFeaturesAcronomy(isSG=isSG, vectorSize=vectorSize, maxNgram=defaultMaxNgram)
            rec_output_file_acronomies_list.append(usedFeaturesAcronym)

        for maxNgram in xrange(1, 11, 1):
            # maxNgram == defaultMaxNgram is already created on the above loop(s)
            if maxNgram == defaultMaxNgram:
                continue
            usedFeaturesAcronym = Common.createUsedFeaturesAcronomy(isSG=isSG, vectorSize=defaultVectorSize, maxNgram=maxNgram)
            rec_output_file_acronomies_list.append(usedFeaturesAcronym)
    return rec_output_file_acronomies_list

def run(selected_users, rec_size, baseRecommendationOutputFileName, rec_output_file_acronomies_list, rec_output_type, recommendationOutputFolderBaseSeq, 
        recommendationOutputFolderBaseSingle, evaluationOutputFilePath):
    
    outFile = open(evaluationOutputFilePath, "a")
    outFile.write(str(rec_output_type)+"\n")
    for rec_output_file_acronomy in rec_output_file_acronomies_list:
        rec_output_file_seq = baseRecommendationOutputFileName +"_sequence" +rec_output_file_acronomy
        rec_output_file = baseRecommendationOutputFileName +"_single" +rec_output_file_acronomy
    
        user_2_rec_seq, user_2_rec_seq_to_single, user_2_rec_seq_to_first = read_rec_seq(recommendationOutputFolderBaseSeq, rec_output_file_seq)
        user_2_rec_single = read_rec_single(recommendationOutputFolderBaseSingle, rec_output_file)
    
        user_2_tp_and_fp_and_prec_and_ndcg = None
        if rec_output_type == "Seq":
            user_2_tp_and_fp_and_prec_and_ndcg = evaluate(rec_output_type, selected_users, rec_size, 
                                                          user_2_list_of_visits_seq, user_2_rec_seq)
        elif rec_output_type == "Seq_Single":
            user_2_tp_and_fp_and_prec_and_ndcg = evaluate(rec_output_type, selected_users, rec_size, 
                                                          user_2_list_of_visits_single, 
                                                          user_2_rec_seq_to_single)
        elif rec_output_type == "Seq_First":
            user_2_tp_and_fp_and_prec_and_ndcg = evaluate(rec_output_type, selected_users, rec_size, 
                                                          user_2_list_of_visits_single, 
                                                          user_2_rec_seq_to_first)
        elif rec_output_type == "Single":
            user_2_tp_and_fp_and_prec_and_ndcg = evaluate(rec_output_type, selected_users, rec_size, 
                                                          user_2_list_of_visits_single, 
                                                          user_2_rec_single)          
                                
        precision_overall = find_overall_precision(user_2_tp_and_fp_and_prec_and_ndcg)
        ndcg_overall = find_overall_ndcg(selected_users,user_2_tp_and_fp_and_prec_and_ndcg)
        hitrate_overall = find_overall_hitrate(selected_users,user_2_tp_and_fp_and_prec_and_ndcg)
        
        
        outFile.write(str(rec_output_file_acronomy)+"\t")
        outFile.write(str(precision_overall)+"\t")
        outFile.write(str(ndcg_overall)+"\t")
        outFile.write(str(hitrate_overall)+"\t")
        outFile.write("\n")
    outFile.close()
        
if __name__ == '__main__':
    dataFolder = "PROJECT_PATH/FastText_Rec/data/checkin2011/checkin2011/checkin2011_janjuly/"
    trainDataFilePath = os.path.join(dataFolder, "user_2_list_of_visits_JanJuly8020_train.csv")
    #test_pruned_output_file_path = "PROJECT_PATH/FastText_Rec/data/checkin2011/checkin2011/checkin2011_janjuly/test_feb_user_2_list_of_visits.csv"
    test_pruned_output_file_path = "PROJECT_PATH/FastText_Rec/data/checkin2011/checkin2011/checkin2011_janjuly/user_2_list_of_visits_JanJuly8020_test_pruned.csv"


    recommendationOutputFolderBase = "PROJECT_PATH/FastText_Rec/output/output_janjuly/fasttext_contentfilter/"
    recommendationOutputFolderBaseSeq =  os.path.join(recommendationOutputFolderBase, "sequence")
    recommendationOutputFolderBaseSingle =  os.path.join(recommendationOutputFolderBase, "single")
    evaluationOutputFilePath = "PROJECT_PATH/FastText_Rec/output/output_janjuly/fasttext_contentfilter/evaluationResult.txt"
    
    baseRecommendationOutputFileName = "fasttext_contentbased_rec"
    rec_size = 10
    

    user_2_list_of_visits_seq, user_2_list_of_visits_single = read_true_visits_seq(test_pruned_output_file_path)
    # select users: who has check-in in both Jan and Feb && whose checkin location in Feb is seen in January
    selected_users = prune_users(user_2_list_of_visits_single, trainDataFilePath)

    rec_output_file_acronomies_list = create_rec_output_file_acronomies_list()
    rec_output_types = ["Seq", "Seq_Single", "Seq_First","Single"]
    for rec_output_type in rec_output_types:
        run(selected_users, rec_size, baseRecommendationOutputFileName, rec_output_file_acronomies_list, rec_output_type, 
            recommendationOutputFolderBaseSeq, recommendationOutputFolderBaseSingle,
            evaluationOutputFilePath)
    print("theend")
    

