# Documentation

## 🔧 Quick Instructions
1. **Configure API Keys**  
   - Open `APIs.csv` and update your API keys

2. **Prepare Papers**  
   - Place PDFs to analyze in `workspace/Papers/`

3. **Run Analysis**  
   - Open `main.py`:
     - Set `query_only = True` to enable claims extractions
     - Write your query in `en.run()`
   - Execute the script
   - *Processing Time:* ~1 hour per 5 papers (progress bars will show status)

4. **Provide Feedback**  
   - After agent generates solution:
     - Enter feedback in input box
     - Or input nothing and press Enter to terminate

## ⚙️ Key Parameters Explained

### Engineer Class
| Parameter        | Type    | Description                                                                 |
|------------------|---------|-----------------------------------------------------------------------------|
| `interactable`   | Boolean | When `True`, agent actively asks questions when facing difficulties/choices |
| `temperature`    | Float   | Controls creativity (0.5-1.5, higher = more creative)                       |
| `memo`           | String  | Pre-knowledge/memories to provide context to the agent                      |

### Librarian.query()
| Parameter             | Type    | Description                                                                   |
|-----------------------|---------|-------------------------------------------------------------------------------|
| `question`            | String  | Your  query                                                                   |
| `items`               | Integer | Number of documents to retrieve, remember, you will receieve the summary only |
| `similarity_threshold`| Float   | Minimum match score (0.0-1.0)                                                 |

## 📁 Files Summary

| File           | Purpose                                                                        |
|----------------|--------------------------------------------------------------------------------|
| `README.md`    | Current documentation file                                                     |
| `Setup.py`     | Configure models and load API keys via `setup()` function                      |
| `main.py`      | Main executable for users to run                                               |
| `format.py`    | Defines structured output formats for data extraction                          |
| `Librarian.py` | Handles:<br>- PDF → JSON extraction<br>- Claim extraction<br>- Query answering |
| `Engineer.py`  | Contains `Engineer` class for idea generation and database access              |
| `APIs.csv`     | Storage for API keys                                                           |

## 📂 Workspace Structure
 ```
 workspace/
 ├── Papers/    # INPUT: Place PDFs here for analysis
 ├── Extracted/ # OUTPUT: Raw text extracted as JSON
 ├── Buffer/    # OUTPUT: Processed claims in JSON format
 └── Database/  # OUTPUT: Vector database for semantic search
 ```

## 🚀 Execution Flow
1. PDFs in `Papers/` → Extracted to JSON in `Extracted/`
2. JSON texts → Processed into claims in `Buffer/`
3. Claims → Vectorized and stored in `Database/`
4. Agent uses database to answer queries



# pip install 'camel-ai[all]' 




