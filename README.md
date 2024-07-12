README
Auction  Price Prediction
Project Description
This project aims to predict the auction sale prices of various types of heavy machinery and construction equipment using historical auction data. The objective is to develop an accurate model that will assist in creating a comprehensive price guide for heavy machinery.

Project Structure
main.py: Main script for data processing, model training, and evaluation.
train.csv: Training data file containing historical auction data including sale prices.
valid.csv: Validation data file containing historical auction data without sale prices.
submission.csv: Submission file containing predicted sale prices for validation data.
requirements.txt: List of dependencies required to run the project.

Data Files
train.csv: Contains the training data with the SalePrice column.
valid.csv: Contains the validation data without the SalePrice column.

Installation
1. Clone the repository:
git clone <repository_url>
cd <repository_directory>

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install the dependencies:
pip install -r requirements.txt

Usage
1. Download the train.csv and valid.csv files from the provided Google Drive links:
Drive links:
train.csv: https://drive.google.com/file/d/1guqSpDv1Q7ZZjSbXMYGbrTvGns0VCyU5/view?usp=drive_link
valid.csv: https://drive.google.com/file/d/1j7x8xhMimKbvW62D-XeDfuRyj9ia636q/view?usp=drive_link

2. Place the downloaded files in the project directory.

3. Run the main script:

python main.py

4. The script will process the data, train the model, and generate predictions for the validation data. 
The predictions will be saved in a file named submission_<timestamp>.csv.

Important Features
The following features were identified as important for predicting auction sale prices:

YearMade: The year the machine was manufactured.
MachineHoursCurrentMeter: Current usage of the machine in hours.
ModelID: Identifier for a unique machine model.
UsageBand_Low: Low usage band compared to average usage for similar machines.
UsageBand_Medium: Medium usage band compared to average usage for similar machines.
fiModelDesc: Description of a unique machine model.
fiBaseModel: Base model of the machine.
fiSecondaryDesc: Secondary description of the model.
fiModelSeries: Series of the model.
fiModelDescriptor: Further disaggregation of the model description.
ProductSize: The size class grouping for a product group.
ProductClassDesc: Description of the second level hierarchical grouping of the model.
Engine_Horsepower: Engine horsepower rating.
Transmission: Type of transmission (automatic or manual).
Drive_System: Machine configuration (e.g., 2 or 4 wheel drive).
Enclosure: Whether the machine has an enclosed cab.
Hydraulics: Type of hydraulics system.
Turbocharged: Whether the engine is naturally aspirated or turbocharged.
Blade_Type: Type of blade.
Undercarriage_Pad_Width: Width of the crawler treads.
Stick_Length: Length of the machine's digging implement.
Coupler: Type of implement interface.
Results and Explanations
The model's performance is evaluated using Root Mean Squared Error (RMSE). Lower RMSE indicates better performance. The script includes hyperparameter tuning, regularization, and feature selection to improve model accuracy and generalization.

License
This project is licensed under the MIT License.

Acknowledgements
Special thanks to the providers of the dataset.
Thanks to the contributors of open-source libraries used in this project.