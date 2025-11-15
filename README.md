# 🚗 Agentic RAG System for Car Information

A simple yet powerful **Retrieval-Augmented Generation (RAG)** system built for freshers to learn AI concepts using **free tools and APIs**.

## 🎯 What You'll Learn
- **RAG Architecture**: How to combine retrieval and generation
- **Vector Databases**: Using FAISS for similarity search
- **LLM Integration**: Working with Groq's free API
- **Web Scraping**: Collecting real-world data
- **Embeddings**: Converting text to vectors

## 🛠️ Tech Stack
- **Groq API**: Free LLaMA 3.1 70B model (30 req/min)
- **FAISS**: Facebook's vector similarity search   
- **SentenceTransformers**: Text embeddings
- **BeautifulSoup**: Web scraping
- **Python**: Simple, beginner-friendly code

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Free Groq API Key
1. Visit: https://console.groq.com/
2. Sign up (completely free)
3. Generate API key

### 3. Run the System
```bash
python car_rag_system.py
```

## 💡 How It Works

```
User Query → Embedding → FAISS Search → Context → Groq LLaMA → Response
```

1. **Data Collection**: Scrapes car info from web
2. **Vectorization**: Converts text to embeddings
3. **Indexing**: Stores in FAISS for fast search
4. **Retrieval**: Finds similar cars for user query
5. **Generation**: Uses Groq to create natural response

## 🎮 Example Interactions

```
You: "Show me electric cars under 20 lakhs"
Assistant: Based on the available data, I found the Tata Nexon EV which is an excellent electric SUV option under 20 lakhs. It's priced at Rs 14.74 - 19.94 Lakh and offers a 465 km range with fast charging capabilities...

You: "Which SUV has the best mileage?"
Assistant: Among the SUVs in the database, the Maruti Suzuki Fronx offers the best mileage at 22.89 kmpl. It's a compact SUV with excellent fuel efficiency...
```

## 📁 Project Structure
```
A project/
├── car_rag_system.py      # Main RAG system
├── requirements.txt       # Dependencies
├── setup_guide.md         # Detailed setup
└── README.md             # This file
```

## 🔧 Key Features
- ✅ **100% Free**: Uses only free APIs and tools
- ✅ **Beginner Friendly**: Simple, well-commented code
- ✅ **Real-time Search**: Fast FAISS vector search
- ✅ **Latest AI**: Groq's LLaMA 3.1 70B model
- ✅ **Interactive**: Chat-based interface
- ✅ **Extensible**: Easy to add more data sources

## 🎓 Learning Path
1. **Start Here**: Run the basic system
2. **Understand**: Read the code comments
3. **Experiment**: Try different queries
4. **Extend**: Add more car data
5. **Customize**: Modify for other domains

## 🆓 Cost Breakdown
- **Groq API**: Free (30 requests/minute)
- **FAISS**: Free (open source)
- **SentenceTransformers**: Free
- **Python Libraries**: Free
- **Total Cost**: $0

Perfect for students and freshers to learn RAG systems without any cost!

cd "C:\Users\mjkat\Documents\TO DO PROJECTS\A RAG CAR"
git init
git add .
git commit -m "Initial commit for A RAG CAR"
# if you have a remote: git remote add origin <url>; git push -u origin main

# VERY IMPORTANT: Back up your work first.
# Then (if you definitely want to remove the Documents-level repo):
rm -Recurse -Force "C:\Users\mjkat\Documents\.git"

# add patterns to .gitignore (edit C:/Users/mjkat/Documents/.gitignore)
git rm -r --cached path/to/dir_or_file
git commit -m "Stop tracking X"