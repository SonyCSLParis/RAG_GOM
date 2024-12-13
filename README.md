# Setup and Launch Guide for Flask and React Project

## Prerequisites

Before starting, make sure you have the following tools installed:
- [Python 3.x](https://www.python.org/downloads/)
- [Node.js](https://nodejs.org/)
- [npm](https://www.npmjs.com/)
- [git](https://git-scm.com/)

## 1. Install Python, Node.js, and npm

Ensure that Python, Node.js, and npm are installed on your machine:

- To check if Python is installed, run:
  ```bash
  python --version
  ```
If not installed, you can download and install Python from the official website.

To check if Node.js and npm are installed, run:
```bash
node --version
npm --version
```
If not installed, you can download and install Node.js (which includes npm) from the official website.

## 2. Set Up a Virtual Environment for Python (Optional but Recommended)

If you want to isolate your Python dependencies, it's recommended to use a virtual environment.

Create a virtual environment:

```bash
python -m venv venv
```
Activate the virtual environment:

On macOS/Linux:

```bash
source venv/bin/activate
```
On Windows (via cmd):

```bash
venv\Scripts\activate
````
If you see the (venv) prefix in your terminal, the virtual environment is activated.

## 3. Install Python Dependencies

With the virtual environment activated (if used), install the required Python packages for your project.

```bash
pip install flask lanchain lanchain-community lanchain-ollama unstructured chromadb
```
## 4. Set the Flask Environment Variables

Now that the virtual environment is set up, navigate to the server directory and set the environment variables for Flask.

Go to the server directory:

```bash
cd server
```
Set the Flask environment variables:

On macOS/Linux:

```bash
export FLASK_APP=server.py
export FLASK_ENV=development
```
On Windows (via cmd):

```bash
set FLASK_APP=server.py
set FLASK_ENV=development
```
To check if the environment variables are set correctly, you can use the echo command (on macOS/Linux) or echo %VAR_NAME% (on Windows).

### 5. Start the Flask Server

Once the environment variables are set, start the Flask server:

```bash
flask run
```
Your Flask application will now be running at http://127.0.0.1:5000.

## 6. Install Front-End Dependencies (React)

Now, let's set up the React part of the project.

Navigate to the front/chatbot directory:

```bash
cd ../front/chatbot
```
Install the necessary Node.js dependencies:

```bash
npm install
```
If the installation fails, try running the command again:

```bash
npm install
```
## 7. Install TailwindCSS

Since TailwindCSS is already configured in your repo, you just need to ensure it's installed.

Install TailwindCSS with npm:

```bash
npm install tailwindcss
```
### 8. Start the React Application

Once all dependencies are installed, launch the React app with:

```bash
npm start
```
The React application will now be running at http://localhost:3000.

### 9. Troubleshooting

If you encounter any issues, here are a few steps to try:

Ensure all dependencies are installed by running npm install and pip install.
Restart your terminal and check that the environment variables are correctly set.

If you encounter issues with Flask, ensure the server.py file exists and is in the correct location.

If you encounter issues with React, check the logs for specific errors and troubleshoot based on the error messages.