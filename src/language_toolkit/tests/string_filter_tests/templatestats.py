# import pandas as pd
# from drain3 import TemplateMiner
# from pprint import pprint
# import spacy 

# if __name__=="__main__":
#     df = pd.read_csv("C:/Users/Zemyna.Mikalciute/LanguageToolkit/src/language_toolkit/tests/data/(CUI).csv", encoding="utf-8")
#     tm = TemplateMiner()

#     for row in df["Message"][1:141]:
#         tm.add_log_message(row)

#         matched_messages = {}


#     unique_temp = set()

#     for row in df["Message"]:
#         t = tm.match(row)
#         if t:
#             matched_messages[row] = t.get_template()  # Store message and its matched template in the dictionary

#     #pprint(unique_temp)
#     pprint(matched_messages)

import pandas as pd
import spacy
from drain3 import TemplateMiner
from pprint import pprint

# Load a language model
nlp = spacy.load("en_core_web_sm")

if __name__=="__main__":
    # Read data from a CSV file
    df = pd.read_csv("C:/Users/Zemyna.Mikalciute/LanguageToolkit/src/language_toolkit/tests/data/(CUI) mm_il4_team_usaf-618aoc-mod_channel_Internal 618 AOC AAD.csv", encoding="utf-8")
 
    # Create a TemplateMiner object
    tm = TemplateMiner()

    # Process the messages using the TemplateMiner object
    matched_messages = {}
    total_processed_messages = 0
    for row in df["message"]:
        if not isinstance(row, (str, bytes)):
          row = str(row)
        tm.add_log_message(row)
        t = tm.match(row)
        if t:
            matched_messages[row] = t.get_template()  # Store message and its matched template in the dictionary
            total_processed_messages += 1

    # Print the matched messages and their templates
    print("Matched Messages and Templates:")
    pprint(matched_messages)

    # Calculate the total number of messages processed
    total_processed_messages = len(matched_messages)

    # Ask the user for the minimum match percentage
    min_match_percentage = float(input("Enter the minimum match percentage: "))

    # Ask the user if they want to extract the features of the templates
    extract_features = input("Do you want to extract the features of the templates? (y/n): ")

    # Loop through each message in the matched_messages dictionary
    template_features = {}
    for message, template in matched_messages.items():
        # Calculate the percentage of messages that conform to this template
        matching_messages = [m for m in matched_messages if matched_messages[m] == template]
        template_percentage = (len(matching_messages) / total_processed_messages) * 100

        # Print the message, template, and percentage only if the percentage is above the minimum match percentage
        if template_percentage >= min_match_percentage:
            print(f"{template_percentage:.2f}% - {message} - {template}")

            # Extract features of the template if requested by the user
            if extract_features.lower() == "y":
                template_features[template] = [word for word in template.split() if "<*>" in word]

    # Print the template features in order of highest percentage to lowest
    if extract_features.lower() == "y":
        sorted_template_features = sorted(template_features.items(), key=lambda x: len(x[1]), reverse=True)
        print("Most common Templates:")
        for template, features in sorted_template_features:
            print(f"{template}: {len(features)} features")


