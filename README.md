# 🎓 AI Quiz Generator

An intelligent quiz generation system powered by T5 and BERT models that creates contextual questions and evaluates answers automatically.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

The AI Quiz Generator is an interactive web application that automatically generates quiz questions from a given dataset using state-of-the-art NLP models. It leverages T5 for question generation and BERT for answer prediction, providing an engaging learning experience with real-time feedback.

## ✨ Features

### Quiz Generation
- 🤖 **AI-Powered Question Generation** using T5 transformer model
- 🎯 **Intelligent Answer Prediction** using BERT model
- 📊 **BLEU Score Evaluation** for question quality assessment
- 🔀 **Randomized Questions** for varied quiz experiences

### Interactive Interface
- 💻 **Beautiful Modern UI** with gradient design and smooth animations
- 📱 **Fully Responsive** - works on desktop, tablet, and mobile
- ⬅️➡️ **Navigate Between Questions** - review and change answers
- 📈 **Real-Time Score Tracking** - see your progress as you answer
- ✅ **Instant Feedback** - immediate validation with correct answers
- 🎉 **Comprehensive Results** - detailed summary with performance metrics

### User Experience
- 🎨 **Elegant Design** with purple gradient theme
- 🔄 **Progress Bar** showing completion status
- 💡 **Context Display** for each question
- 🏆 **Performance Messages** - encouraging feedback based on score
- 📝 **Answer Review** - see all questions and answers at the end

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│   (HTML/CSS/JS) │
└────────┬────────┘
         │
    HTTP Requests
         │
┌────────▼────────┐
│   Flask API     │
│   (Backend)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼───┐
│  T5   │ │ BERT │
│ Model │ │ Model│
└───────┘ └──────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Colab account (for deployment)
- ngrok account (for public URL)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ai-quiz-generator.git
cd ai-quiz-generator
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Additional Requirements

```bash
pip install flask transformers torch pandas numpy nltk pyngrok
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('punkt')
```

## 📖 Usage

### Running Locally

1. **Prepare Your Dataset**
   ```python
   import pandas as pd
   
   # Load your dataset
   df = pd.read_csv('your_dataset.csv')
   # Ensure it has columns: context, question, answer
   ```

2. **Initialize Models**
   ```python
   from transformers import T5Tokenizer, T5ForConditionalGeneration
   from transformers import pipeline
   
   # Load T5 model for question generation
   t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
   t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
   
   # Load BERT model for answer prediction
   bert_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
   ```

3. **Define Quiz Generation Function**
   ```python
   def generate_quiz_t5_bert(df, num_questions=5):
       # Your quiz generation logic here
       pass
   ```

4. **Run Flask App**
   ```python
   python app.py
   ```

### Running on Google Colab

1. **Upload Notebook to Colab**
   - Open Google Colab
   - Upload your notebook file

2. **Install pyngrok**
   ```python
   !pip install pyngrok -q
   ```

3. **Set ngrok Auth Token**
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_TOKEN")
   ```

4. **Run the Application**
   - Execute all cells in order
   - Access the public URL provided by ngrok

### Using the Interface

1. **Start Quiz**
   - Enter the number of questions (1-20)
   - Click "Start Quiz"

2. **Answer Questions**
   - Read the context and question
   - Type your answer in the text box
   - Press Enter or click "Submit Answer"

3. **Navigate**
   - Use "← Previous" to review earlier questions
   - Use "Next Question →" to move forward
   - Click "Finish Quiz" on the last question

4. **Review Results**
   - See your final score and percentage
   - Review all questions with correct answers
   - Click "Take Another Quiz" to restart

## 🛠️ Technologies Used

### Backend
- **Flask** - Web framework
- **Transformers** - Hugging Face library for NLP models
- **PyTorch** - Deep learning framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **NLTK** - Natural language toolkit

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with gradients and animations
- **JavaScript (ES6+)** - Interactive functionality
- **Fetch API** - Asynchronous requests

### Models
- **T5 (Text-to-Text Transfer Transformer)** - Question generation
- **BERT (Bidirectional Encoder Representations from Transformers)** - Answer prediction

### Deployment
- **ngrok** - Secure tunneling for public URLs
- **Google Colab** - Cloud-based notebook environment

## 🔍 How It Works

### 1. Question Generation (T5)
```python
# T5 generates questions from context
input_text = f"generate question: {context}"
question = t5_model.generate(input_text)
```

### 2. Answer Prediction (BERT)
```python
# BERT predicts answers from context and question
result = bert_pipeline(question=question, context=context)
predicted_answer = result['answer']
confidence = result['score']
```

### 3. Quality Evaluation (BLEU)
```python
# Calculate similarity between generated and reference questions
from nltk.translate.bleu_score import sentence_bleu
bleu_score = sentence_bleu([reference], generated)
```

### 4. Quiz Flow
1. User selects number of questions
2. Backend generates questions using T5
3. Questions are randomized
4. User answers each question
5. Backend validates answers against BERT predictions
6. Final score and detailed results are displayed

## ⚙️ Configuration

### Model Configuration

```python
# config.py
T5_MODEL = 't5-small'  # Options: t5-small, t5-base, t5-large
BERT_MODEL = 'bert-large-uncased-whole-word-masking-finetuned-squad'
MAX_QUESTIONS = 20
MIN_QUESTIONS = 1
```

### Flask Configuration

```python
# app.py
app.config['DEBUG'] = False
app.config['PORT'] = 5000
app.config['HOST'] = '0.0.0.0'
```

### ngrok Configuration

```python
# Get your token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "your_token_here"
```

## 🐛 Troubleshooting

### Common Issues

**Issue: "Site cannot be reached"**
- Solution: Click through the ngrok warning page and click "Visit Site"

**Issue: "Port 5000 already in use"**
```python
# Change the port number
app.run(host='0.0.0.0', port=5001)
ngrok.connect(5001)
```

**Issue: "Model loading timeout"**
- Solution: Use smaller models (t5-small instead of t5-large)
- Increase Colab RAM: Runtime → Change runtime type → High-RAM

**Issue: "generate_quiz_t5_bert not defined"**
- Solution: Run all previous cells in order before starting the Flask app

**Issue: "DataFrame not found"**
- Solution: Ensure your dataset is loaded before calling the generation function

### Memory Issues

If you encounter memory errors:
1. Reduce batch size
2. Use smaller models
3. Process fewer questions at a time
4. Restart Colab runtime and clear outputs

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** - For providing pre-trained transformer models
- **Google Research** - For T5 and BERT models
- **Flask Community** - For the excellent web framework
- **ngrok** - For secure tunneling solution

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/ai-quiz-generator](https://github.com/yourusername/ai-quiz-generator)

## 📊 Performance Metrics

- **Question Generation Time**: ~2-5 seconds per question
- **Answer Prediction Time**: ~1-2 seconds per answer
- **BLEU Score Range**: 0.0 - 1.0 (higher is better)
- **Typical Confidence**: 0.3 - 0.9 (BERT answer confidence)

## 🎯 Future Enhancements

- [ ] Multiple choice question generation
- [ ] Different difficulty levels
- [ ] Support for multiple languages
- [ ] User authentication and progress tracking
- [ ] Export quiz results as PDF
- [ ] Integration with learning management systems
- [ ] Custom dataset upload functionality
- [ ] Advanced analytics and reporting
- [ ] Mobile app version
- [ ] Voice-based quiz interface

## 📚 Resources

- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [ngrok Documentation](https://ngrok.com/docs)

---

Made with ❤️ and 🤖 by [Your Name]
