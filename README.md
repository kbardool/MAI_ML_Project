# classifydotbe
Machine Learning Project files for classification of Belgian websites.

## Requirements:
- mongoDB (ver >= 3.2) listening on port 27017
- python 3.5
- packages in requirements.txt 

### Installation using Conda:
Use the requirements.txt file in conjunction with Conda to install all dependencies

- copy requirements.txt file
- $ conda create --name <env> --file requirements.txt

### Execution 

  $ classifydotbe input_file.csv output_file.csv
  
  Executes the classifydotbe.bat command file taking input_file and output_file parms 
  
  - input_file: Input file in CSV format (assumes the first line is a header line)
  - output_file : Classified file in CSV format
  
  Example:
  
  classifydotbe  input_file100.csv  output.csv
  
  There are a few test csv files in the main project folder
  
  - input10.csv, input40.csv, input100.csv  
  
  Input files with 10,40, and 100 domain names respectively.

  ** please note that the loader program will skip over the header in the input CSV file.
  
  
  Thank you.
