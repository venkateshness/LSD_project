#%%
import pandas as pd
import os
import re

HOMEDIR = "/Brain/private/v20subra/LSD_project/"

ID = pd.read_excel(f"{HOMEDIR}/src_data/IDs.xlsx", header=2)
filtered_ID = ID[["ID", ID.columns[-1]]]
filtered_ID['ID'] = filtered_ID['ID'].str.replace('_', '-')
filtered_ID.columns = ['ID', 'subject #']
LSD_PLA_order = pd.read_csv(f"{HOMEDIR}/src_data/LSD_and_PLA_Data.csv", delimiter=";").dropna()

df_all = pd.merge(LSD_PLA_order, filtered_ID, on ="subject #", how="inner").dropna()

# df_all.to_excel(f'{HOMEDIR}/src_data/LSD_PLA_order_ID.xlsx')
#%%
df_all

# %%
# rename folder 010813-4_LSD_20140820_Video to 010814-4_LSD_20140820_Video

# Define the base directory containing the mislabelled folders and files
base_directory = f"{HOMEDIR}/src_data/ds_data_batch2/Music/"

# Define the pattern that needs correction (e.g., incorrect label)
pattern = r'010814-4_LSD_20140903_Closed1'  # Replace with the actual pattern
replacement = '010813-4_LSD_20140903_Closed1'  # Replace with the correct label

def rename_files_and_dirs(base_dir, pattern, replacement):
    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Walk through the directory and subdirectories (bottom-up)
    for root, dirs, files in os.walk(base_dir, topdown=False):
        # Rename files
        for filename in files:
            if regex.search(filename):
                old_path = os.path.join(root, filename)
                new_filename = regex.sub(replacement, filename)
                new_path = os.path.join(root, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f'Renamed file: {old_path} -> {new_path}')
                except Exception as e:
                    print(f'Error renaming file {old_path}: {e}')

        # Rename directories
        for dirname in dirs:
            if regex.search(dirname):
                old_path = os.path.join(root, dirname)
                new_dirname = regex.sub(replacement, dirname)
                new_path = os.path.join(root, new_dirname)

                try:
                    os.rename(old_path, new_path)
                    print(f'Renamed directory: {old_path} -> {new_path}')
                except Exception as e:
                    print(f'Error renaming directory {old_path}: {e}')


rename_files_and_dirs(base_directory, pattern, replacement)

pattern = r'Closed1'
replacement = 'Closed' 
rename_files_and_dirs(base_directory, pattern, replacement)
# %%



# File starts with
# sort, get the order
# look up excel; order, condition
# slap LSD or PLAC


directory=f"{HOMEDIR}/src_data/ds_data_batch2/Music"

# listdir, sort order
def startswith_and_order(subID):
    files_with_prefix = [f for f in sorted(os.listdir(directory)) if f.startswith(f"{subID}")]
    
    return files_with_prefix

dict_by_subj = {}
for subID in  (df_all[df_all['Condition']=="LSD"]['ID'].values):
    ordered = startswith_and_order(subID)
    if len(ordered)>0:
        dict_by_subj[subID] = ordered

new_labelling = []
for key in dict_by_subj.keys():

# look up the excel
    order_for_subjects = df_all[df_all['ID']==f'{key}']['Order_LSD'].values
    
# look for the order for condition LSD; 
# if it's 1, that means LSD session is #1 we have to write PLA to file #2
    if order_for_subjects[0] == '1':
        print(dict_by_subj[f'{key}'])
        new_labelling.append(dict_by_subj[f'{key}'][0].replace('LSD', 'LSD'))
        new_labelling.append(dict_by_subj[f'{key}'][1].replace('LSD', 'PLA'))

# if it's 2, that means LSD session is #2 we have to write PLA to file #1
    if order_for_subjects[0] == '2':
        print(dict_by_subj[f'{key}'])
        new_labelling.append(dict_by_subj[f'{key}'][0].replace('LSD', 'PLA'))
        new_labelling.append(dict_by_subj[f'{key}'][1].replace('LSD', 'LSD'))


#%%
dict_by_subj
#%%
original_folders = [f for f in sorted(os.listdir(directory)) if f.endswith(f"Music.ds")]


for original, new in zip(sorted(original_folders), sorted(new_labelling)):
    old_path = os.path.join(directory, original)
    new_path = os.path.join(directory, new)
    
    if original not in new_labelling:
        new = original.replace('LSD', 'PLA')
        new_path = os.path.join(directory, new)
        os.rename(old_path, new_path)
        
#%%
#first put LSD and PLA folders into their their respective directory
#### replacing LSD to PLA for sub folders in the PLA directory
import os
subjects_list = os.listdir(f"{HOMEDIR}/src_data/ds_data_batch2/Music/PLA/")

for eachsubject in subjects_list:
    directory_to_search = os.listdir(f'{HOMEDIR}/src_data/ds_data_batch2/Music/PLA/{eachsubject}')
    for old in directory_to_search:

        directory = f'{HOMEDIR}/src_data/ds_data_batch2/Music/PLA/{eachsubject}/'
        old_path = os.path.join(directory, old)

        new = old.replace('LSD', 'PLA')
        new_path = os.path.join(directory, new)
        os.rename(old_path, new_path)

# # %%

# %%

pd.read_excel(f'{HOMEDIR}/src_data/IDs.xlsx', header=2)
# %%
