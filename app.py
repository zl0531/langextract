import os
import sys
import json
import langextract as lx
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
import uuid

# Text extraction imports
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 is not installed. Please install it using: pip install PyPDF2")
    sys.exit(1)

try:
    import docx
except ImportError:
    print("python-docx is not installed. Please install it using: pip install python-docx")
    sys.exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///langextract.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Models ---
class Example(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    text = db.Column(db.Text, nullable=False)
    extractions = db.Column(db.Text, nullable=False)  # Storing extractions as JSON string

    def __repr__(self):
        return f'<Example {self.name}>'

class JobHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_name = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    result = db.Column(db.Text, nullable=True) # Made nullable
    result_path = db.Column(db.String(200), nullable=True) # Added and made nullable

    def __repr__(self):
        return f'<JobHistory {self.id}>'

# --- Text Extraction Functions ---
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file, including tables."""
    doc = docx.Document(docx_path)
    text = ""
    for block in doc.iter_inner_content(): # Iterate through all block-level content
        if isinstance(block, docx.text.paragraph.Paragraph):
            text += block.text + '\n'
        elif isinstance(block, docx.table.Table):
            text += "\n--- TABLE START ---\n"
            for row in block.rows:
                row_text = []
                for cell in row.cells:
                    row_text.append(cell.text)
                text += " | ".join(row_text) + "\n"
            text += "--- TABLE END ---\n\n"
    return text

# --- Routes ---
@app.route('/')
def index():
    job_id = request.args.get('job_id', type=int)
    visualization_html = None
    if job_id:
        job = JobHistory.query.get(job_id)
        if job and job.result_path:
            # lx.visualize expects a path to a JSONL file
            visualization_html = lx.visualize(job.result_path)

    history = JobHistory.query.order_by(JobHistory.timestamp.desc()).all()
    examples = Example.query.all()
    return render_template(
        'index.html',
        history=history,
        examples=examples,
        visualization_html=visualization_html
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'filename': filename})
    return redirect(url_for('index'))

@app.route('/extract', methods=['POST'])
def extract():
    data = request.form
    filename = data['filename']
    example_ids = [int(id_str) for id_str in json.loads(data['example_ids'])]
    api_key = data['api_key']

    # Get examples from DB
    selected_examples = Example.query.filter(Example.id.in_(example_ids)).all()
    examples_for_lx = []
    for e in selected_examples:
        extractions_json = json.loads(e.extractions)
        extractions_objects = [
            lx.data.Extraction(**extraction_dict)
            for extraction_dict in extractions_json
        ]
        examples_for_lx.append(
            lx.data.ExampleData(
                text=e.text,
                extractions=extractions_objects
            )
        )

    # Extract text from file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if file_path.lower().endswith('.pdf'):
        input_text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        input_text = extract_text_from_docx(file_path)
    else:
        return jsonify({'error': 'Unsupported file format.'}), 400

    # Run extraction
    with open('prompt.md', 'r', encoding='utf-8') as f:
        prompt = f.read()

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples_for_lx,
        model_id="gemini-2.5-flash", # Corrected model ID
        api_key=api_key,
    )

    # Save job history
    result_filename = f'{uuid.uuid4()}.jsonl'
    output_dir = 'results'
    lx.io.save_annotated_documents([result], output_name=result_filename, output_dir=output_dir)
    result_path = os.path.join(output_dir, result_filename)

    new_job = JobHistory(
        document_name=filename,
        result="", # Provide empty string for legacy 'result' column
        result_path=result_path
    )
    db.session.add(new_job)
    db.session.commit()

    return redirect(url_for('index', job_id=new_job.id))

@app.route('/example', methods=['POST'])
def add_example():
    data = request.form.to_dict()
    if 'extractions' not in data:
        history = JobHistory.query.order_by(JobHistory.timestamp.desc()).all()
        examples = Example.query.all()
        return render_template(
            'index.html',
            history=history,
            examples=examples,
            form_error='The "Extractions (JSON)" field is missing.',
            form_data=data
        )
    try:
        # Validate JSON format
        json.loads(data['extractions'])
    except json.JSONDecodeError as e:
        history = JobHistory.query.order_by(JobHistory.timestamp.desc()).all()
        examples = Example.query.all()
        return render_template(
            'index.html',
            history=history,
            examples=examples,
            form_error=f'Invalid JSON format in Extractions: {e}',
            form_data=data
        )

    new_example = Example(
        name=data['name'],
        category=data['category'],
        text=data['text'],
        extractions=data['extractions'] # This is a JSON string
    )
    db.session.add(new_example)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/examples/<int:example_id>')
def get_example_details(example_id):
    example = Example.query.get(example_id)
    if example:
        return jsonify({
            'id': example.id,
            'name': example.name,
            'category': example.category,
            'text': example.text,
            'extractions': example.extractions
        })
    return jsonify({'error': 'Example not found'}), 404

@app.route('/examples/<category>')
def get_examples_by_category(category):
    examples = Example.query.filter_by(category=category).all()
    return jsonify([{'id': e.id, 'name': e.name} for e in examples])

@app.route('/examples/categories')
def get_example_categories():
    categories = db.session.query(Example.category).distinct().all()
    return jsonify([c[0] for c in categories])

@app.route('/example/update/<int:example_id>', methods=['POST'])
def update_example(example_id):
    example = Example.query.get(example_id)
    if not example:
        return jsonify({'error': 'Example not found'}), 404

    data = request.get_json()
    try:
        json.loads(data['extractions'])
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON format in Extractions: {e}'}), 400

    example.name = data['name']
    example.category = data['category']
    example.text = data['text']
    example.extractions = data['extractions']
    db.session.commit()
    return jsonify({'message': 'Example updated successfully'})

def init_db():
    with app.app_context():
        db.create_all()
        # Add result_path column to JobHistory table if it doesn't exist
        inspector = inspect(db.engine)
        columns = [c['name'] for c in inspector.get_columns('job_history')]
        if 'result_path' not in columns:
            with db.engine.connect() as con:
                con.execute(text('ALTER TABLE job_history ADD COLUMN result_path VARCHAR(200)'))
                con.commit()
        # Make 'result' column nullable if it's not already
        # This is a more complex migration, typically handled by Alembic.
        # For simplicity, we'll assume it's nullable after the initial create_all
        # or that the user has deleted the db. If not, manual intervention might be needed.
        # If 'result' column exists and is NOT NULL, this would require recreating the table.
        # For now, we rely on the model definition having nullable=True.


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', debug=True)