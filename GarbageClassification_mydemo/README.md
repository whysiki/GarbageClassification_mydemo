# Laji Project

## Overview
The Laji Project is an intelligent waste classification system that utilizes image detection and sensor technology to identify and categorize different types of waste. The application features a user-friendly interface built with PySide6, allowing users to interact with the system seamlessly.

## File Structure
```
laji_project
├── src
│   ├── main.py            # Main entry point of the application
│   ├── ui_layout.py       # Defines the user interface layout
│   └── detect_moudle.py   # Contains the detection module and sensor operations
├── requirements.txt       # Lists the dependencies required for the project
└── README.md              # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd laji_project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/main.py
   ```

## Usage
- Upon launching the application, users can start the waste classification process by clicking the "开始" (Start) button.
- The system will utilize the camera to detect waste items and classify them into categories such as "可回收垃圾" (Recyclable), "厨余垃圾" (Kitchen Waste), "有害垃圾" (Hazardous Waste), and "其他垃圾" (Other Waste).
- Users can view the detection results and logs in the application interface.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License.