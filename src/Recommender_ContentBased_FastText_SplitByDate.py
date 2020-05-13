'''
Created on Mar, 2018

@author: mgulcin
'''

from gensim.models.fasttext import FastText as FT_gensim
import os
import time

import Common


def printRecommendationForASingleUser(recommendationOutputFilePath, user_id, recommendations):
    outFile = open(recommendationOutputFilePath, "a")
    outFile.write(str(user_id)+",")
    #updated_recommendations = ("loc"+s for s in recommendations)
    #outFile.write(str(','.join(updated_recommendations)))
    outFile.write(str(','.join(recommendations)))
    outFile.write("\n")
    outFile.close()

def findSimScorePerItem(most_similar_items_and_sim_score):
    # most_similar_item_and_score is like: (loc1_loc2_..., 0.99)
    # convert to: loc1->[0.99], loc2->[0.99]
    item_2_scores = {}
    for most_similar_item_and_score in most_similar_items_and_sim_score:
        items = most_similar_item_and_score[0].split('_')
        score = most_similar_item_and_score[1]
        for item in items:
            old_scores_list = []
            if item in item_2_scores:
                old_scores_list = item_2_scores[item]
            old_scores_list.append(score)
            item_2_scores[item] = old_scores_list
    return item_2_scores

def rankRecommendations(item_2_scores):
    item_2_avg_score = {}
    for item, scores in item_2_scores.items():
        item_2_avg_score[item] = sum(scores) / float(len(scores))
    sorted_item_and_avg_score = sorted(item_2_avg_score.items(), key=lambda x: x[1], reverse=True)
    recommendations = [item_and_avg_score[0] for item_and_avg_score in sorted_item_and_avg_score]
    return recommendations

def  decideRecommendations_Single(most_similar_items_and_sim_score):
    # most_similar_item_and_score is like: (loc1_loc2_..., 0.99)
    # convert to: loc1->[0.99], loc2->[0.99]
    item_2_scores = findSimScorePerItem(most_similar_items_and_sim_score)
    # find avg. scores and recommend ordered (ranked) items 
    # TODO other scoring functions can be applied!! 
    recommendations = rankRecommendations(item_2_scores)
    return recommendations

def  decideRecommendations_Sequence(most_similar_items_and_sim_score):
    # most_similar_item_and_score is like, a tuple: (loc1_loc2_..., 0.99)
    # convert to, the key: loc1_loc2
    recommendations = [most_similar_item_and_sim_score[0] for most_similar_item_and_sim_score in most_similar_items_and_sim_score]
    return recommendations

def decideRecommendations(recommendation_type, most_similar_items_and_sim_score, rec_output_size):
    recommendations = []
    if recommendation_type == "SEQUENCE":
        recommendations = decideRecommendations_Sequence(most_similar_items_and_sim_score)
    elif recommendation_type == "SINGLE":
        recommendations = decideRecommendations_Single(most_similar_items_and_sim_score)
    return recommendations[0:rec_output_size]

def recommend(recommendation_type, rec_output_size,
              loaded_model, user_2_locations, recommendationOutputFilePath):
    for user_id, user_locations in user_2_locations.items():
        # find the top-k most similar items(loc.) to the already visited locations
        most_similar_items_and_sim_score= loaded_model.most_similar(user_locations, topn=rec_output_size)
        recommendations = decideRecommendations(recommendation_type, most_similar_items_and_sim_score, rec_output_size)
        #write the output to the file
        printRecommendationForASingleUser(recommendationOutputFilePath, user_id, recommendations)
        
def readModelAndMakeRecommendation(recommendation_type, rec_output_size,
                                   modelFolder, baseModelFileName, 
                                   timeInfoFolder, timeInfoFile, 
                                   recommendationOutputFolder, baseRecommendationOutputFileName,
                                   isSG, vectorSize, maxNgram, user_2_locations):
    start_time = time.time()
    usedFeaturesAcronym = Common.createUsedFeaturesAcronomy(isSG=isSG, vectorSize=vectorSize, maxNgram=maxNgram)
    modelFilePath = os.path.join(modelFolder, baseModelFileName + usedFeaturesAcronym)
    loaded_model = FT_gensim.load(modelFilePath)
    
    recommendationOutputFilePath = os.path.join(recommendationOutputFolder, baseRecommendationOutputFileName + usedFeaturesAcronym)
    recommend(recommendation_type,  rec_output_size, loaded_model, user_2_locations, recommendationOutputFilePath)
    Common.printTime((time.time() - start_time), usedFeaturesAcronym, timeInfoFolder, timeInfoFile)
   
   
def run(recommendation_type, rec_output_size,
        trainDataFilePath, test_pruned_output_file_path, modelFolder, baseModelFileName, timeInfoFolder, timeInfoFile,
        recommendationOutputFolder, baseRecommendationOutputFileName):
    # read the training data
    train_user_2_locations = Common.readTrainData(trainDataFilePath)
    test_user_2_locations = Common.readTrainData(test_pruned_output_file_path)

    user_2_locations = train_user_2_locations

    # add first location set seen in test time!!
    for user, loc in test_user_2_locations.items():
        if user in user_2_locations:
            # already have some locations
            train_locations = user_2_locations[user]
            if len(loc) < 1:
                print("user with no checkin at test: {}".format(user))
                continue
            #train_locations.extend(loc[:-1]) # all but last
            user_2_locations[user] = loc[0]# use first
        else:
            # new user in test time
            user_2_locations[user] = loc[0] # use first

    # read model and make rec.
    vectorSizes = [10, 50, 100, 150, 200, 250]
    isSGValues = [0, 1]
    maxngram_values = [5,9]#range(0, 11, 1)
    defaultVectorSize = 100
    for isSG in isSGValues:
        # for vectorSize in vectorSizes:
        #     readModelAndMakeRecommendation(recommendation_type=recommendation_type, rec_output_size=rec_output_size,
        #                                    modelFolder=modelFolder, baseModelFileName=baseModelFileName,
        #                                    timeInfoFolder=timeInfoFolder, timeInfoFile=timeInfoFile,
        #                                    recommendationOutputFolder=recommendationOutputFolder,
        #                                    baseRecommendationOutputFileName=baseRecommendationOutputFileName,
        #                                    isSG=isSG, vectorSize=vectorSize, maxNgram=defaultMaxNgram,
        #                                    user_2_locations=user_2_locations)

        for maxNgram in maxngram_values:
            # maxNgram == defaultMaxNgram is already created on the above loop(s)
            # if maxNgram == defaultMaxNgram:
            #     continue
            readModelAndMakeRecommendation(recommendation_type=recommendation_type, rec_output_size=rec_output_size,
                                           modelFolder=modelFolder, baseModelFileName=baseModelFileName, 
                                           timeInfoFolder=timeInfoFolder, timeInfoFile=timeInfoFile,
                                           recommendationOutputFolder=recommendationOutputFolder,
                                           baseRecommendationOutputFileName=baseRecommendationOutputFileName,
                                           isSG=isSG, vectorSize=defaultVectorSize, maxNgram=maxNgram, 
                                           user_2_locations=user_2_locations)
    return

if __name__ == '__main__':

    base_dir = "XXX"
    dataFolder = "XXX/dataset/checkins2011_janjuly/"
    trainDataFilePath = os.path.join(dataFolder, "user_2_list_of_visits_2011_05_01_train.csv")
    test_pruned_output_file_path = dataFolder + "/user_2_list_of_visits_2011_05_01_test_pruned.csv"
    modelFolder = base_dir + "model/model_janjuly/model_fasttext/"
    baseModelFileName = "fasttext_locs_only_model"
    timeInfoFolderBase = base_dir + "/output/output_janjuly/fasttext_contentfilter"
    recommendationOutputFolderBase = base_dir + "/output/output_janjuly/fasttext_contentfilter/"


    rec_output_size = 10 #rec. size
    recommendation_types = ["SEQUENCE", "SINGLE"] #TODO convert to enum
    for recommendation_type in recommendation_types:
        timeInfoFile = "time_fasttext_contentbased_rec_"+recommendation_type.lower()+".txt"
        timeInfoFolder = os.path.join(timeInfoFolderBase, recommendation_type.lower())
        baseRecommendationOutputFileName = "fasttext_contentbased_rec_"+recommendation_type.lower()
        recommendationOutputFolder = os.path.join(recommendationOutputFolderBase, recommendation_type.lower())
        if not os.path.exists(timeInfoFolder):
            os.makedirs(timeInfoFolder)
        if not os.path.exists(recommendationOutputFolder):
            os.makedirs(recommendationOutputFolder)
    
        run(recommendation_type, rec_output_size,
            trainDataFilePath,  test_pruned_output_file_path, modelFolder, baseModelFileName,
            timeInfoFolder, timeInfoFile, recommendationOutputFolder, baseRecommendationOutputFileName)
    
     
    
    