from flask import Flask, render_template, request
from solve_code.execute import main
import numpy as np
import os
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def process_image(filename):
    # Process the image and generate the Sudoku board
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    sudoku_board = main(file_path)
    return sudoku_board

@app.route('/')
def index():
    return render_template('index.html')

# Handle POST requests in the form of uploaded images.
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        if file:
            filename = file.filename
            file.save(f"{app.config['UPLOAD_FOLDER']}/{filename}")
            
            sudoku_board = process_image(filename) # Process image and generate solution

            shutil.rmtree(app.config['UPLOAD_FOLDER'])  # Remove the 'uploads' directory to clean up

            return render_template('index.html', sudoku_board=sudoku_board) # Display solution
        
        return render_template('index.html')
    except Exception as e:
        error_message = f"Something went wrong: {str(e)}"
        return render_template('index.html', error=error_message)
    
if __name__ == '__main__':
    app.run(debug=True, port=8080)
