'''
Created on Mar, 2018

@author: mgulcin
'''

import csv
import os


def readTrainData(filename):
    user_2_locations = {}
    # read document     
    with open(filename, "r") as inFile:
        reader = csv.reader(inFile, delimiter='\t')

        # populate list (each line in the csv is a list), whole input is a list of list
        # input is like: userid \t [locid1,locid2] \t locid23 \t [locid2,locid27,...] \t ...
        # convert to: locid1_locid2 \t locid23 \t locid2_locid27_... \t ...
        for row in reader:
            # remove [, ]
            updated_row = [value.replace("[", "") for value in row]
            updated_row = [value.replace("]", "") for value in updated_row]
            # use _ as the separator
            updated_row = [value.replace(", ", "_") for value in updated_row]

            user_id = updated_row[0]
            user_locations = updated_row[1:]
            user_2_locations[user_id] = user_locations
    return user_2_locations

def readTrainDataNonSeq(trainDataFileName):
    
    # read document     
    inFile =  open(trainDataFileName, "r")
    reader = csv.reader(inFile, delimiter='\t')
    
    # populate list (each line in the csv is a list), whole input is a list of list
    # input is like: userid \t [locid1,locid2] \t locid23 \t [locid2,locid27,...] \t ...
    # convert to: locid1 \t locid2 \t locid23 \t locid2 \t locid27 ... \t ...
    user_2_locations = {}
    for row in reader:
        # remove [, ]
        updated_row = [value.replace("[", "") for value in row]
        updated_row = [value.replace("]", "") for value in updated_row]
        # use _ as the separator
        updated_row = [value.replace(", ", "_") for value in updated_row]
        
        user_id = updated_row[0]
        user_locations = updated_row[1:]
        user_locations_flattened = [item_list for sublist in user_locations for item_list in sublist.split("_")]
        user_2_locations[user_id] = user_locations_flattened
    return user_2_locations

def createUsedFeaturesAcronomy(isSG, vectorSize, maxNgram):
    usedFeaturesAcronym = ""
    if(isSG == 1):
        usedFeaturesAcronym="_SkipGram"
    elif(isSG == 0):
        usedFeaturesAcronym="_CBow"
    else:
        print("Wrong type_isSG")
    usedFeaturesAcronym = usedFeaturesAcronym + "_VS=" + str(vectorSize) + "_MaxN="+str(maxNgram)
    return usedFeaturesAcronym

def createInvertedIndex(user_2_locations):
    ## e.g. userid --> locid1_locid2 \t locid23 \t locid2_locid27_... \t ...
    location_2_users ={}
    for user, locations in user_2_locations.iteritems():
        for location_grp in locations:
            splitted_locations = location_grp.split('_')
            for location in splitted_locations:
                if location in location_2_users:
                    users = location_2_users[location]
                    users.add(user)
                    location_2_users[location] = users
                else:
                    location_2_users[location] = {user}
        
    return location_2_users
                            
def printTime(timeValue, timeExplanation, timeInfoFolder, timeInfoFile):
    timeInfoFilePath = os.path.join(os.path.dirname(__file__), timeInfoFolder, timeInfoFile)
    outFile = open(timeInfoFilePath, "a")
    outFile.write(timeExplanation+",")
    outFile.write(str(timeValue))
    outFile.write("\n")


