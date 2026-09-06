from pathlib import Path
import requests


# create a parent folder called data
data_path = Path("data/") 

# create a sub folder called the-verdict within the data folder
text_data_path = data_path / "the-verdict"

# if the folder does not exist, create it
if  text_data_path.is_dir():
    print(f"the {text_data_path} laready exist")
else:
    print(f"the {text_data_path} does not exist")
    text_data_path.mkdir(parents=True, exist_ok=True) 

# create a file called the-verdict.txt within the the-verdict folder
text_data_file_path = Path("the-verdict.txt")

if text_data_file_path.is_file():
  print('the data already exist within the txt file')
else:
  # we should now open the file path that we have created for saving teh text file 
  with open(text_data_path / "the-verdict.txt", 'wb') as f:
      request = requests.get("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
      print('we are downloading the data')
      f.write(request.content)
