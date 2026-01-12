import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generation.rag_chain import LegalRAGChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_golden_dataset():
    dataset_path = PROJECT_ROOT / "tests" / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_validation_report():
    logger.info("Starting End-to-End Validation (Retrieval + Generation)...")
    
    # Load dataset
    data = load_golden_dataset()
    questions = data["questions"]
    
    # Initialize RAG Chain
    try:
        rag_chain = LegalRAGChain()
    except Exception as e:
        logger.error(f"Failed to initialize RAG Chain: {e}")
        return

    report_lines = []
    report_lines.append("# Malaysian Legal RAG - Validation Report")
    report_lines.append("Comparing generated answers against ground truth.\n")
    
    results = []

    # Process each question
    for i, item in enumerate(tqdm(questions, desc="Validating")):
        q_id = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_act = item["expected_act"]
        expected_section = item["expected_section"]
        
        logger.info(f"Processing {q_id}: {question}")
        
        try:
            # Run RAG Pipeline
            response = rag_chain.ask(question, return_sources=True)
            generated_answer = response["answer"]
            sources = response["sources"]
            
            # success check
            retrieved_correctly = False
            retrieved_sections = []
            
            for s in sources:
                s_act = s.act_name if hasattr(s, 'act_name') else s.get('act_name')
                s_sec = s.section_number if hasattr(s, 'section_number') else s.get('section_number')
                retrieved_sections.append(f"{s_act} (Sec {s_sec})")
                
                # Check for match
                if expected_act in s_act and expected_section.replace("Section ", "") in str(s_sec):
                    retrieved_correctly = True

            # Append to report
            report_lines.append(f"## {q_id}: {question}")
            report_lines.append(f"**Status**: {'✅ Retrieval Success' if retrieved_correctly else '❌ Retrieval Failed'}")
            report_lines.append(f"\n**Expected Source**: {expected_act}, {expected_section}")
            report_lines.append(f"**Retrieved Sources**: {', '.join(retrieved_sections[:3])}")
            
            report_lines.append("\n### Ground Truth Answer")
            report_lines.append(f"> {ground_truth}")
            
            report_lines.append("\n### Generated Answer")
            report_lines.append(generated_answer)
            report_lines.append("\n---")
            
        except Exception as e:
            logger.error(f"Error processing {q_id}: {e}")
            report_lines.append(f"## {q_id}: {question}") 
            report_lines.append(f"\n**ERROR**: {str(e)}")
            report_lines.append("\n---")
            
    # Save Report
    output_path = PROJECT_ROOT / "tests" / "validation_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Validation complete. Report saved to {output_path}")

if __name__ == "__main__":
    generate_validation_report()
