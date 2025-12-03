### **🧩 Challenge: Context-Aware Question Answering under Token Constraints**

#### ***Objective***

Your mission is to design a **question-answering (QA) system** that intelligently selects and presents the most relevant context to a **local language model** (LLM) under a **strict token budget**.
 The LLM can only “see” a limited portion of the knowledge base, so your system must make optimal decisions about which pieces of information to include.

#### ***🎯 Interview Objective***

During the discussion, we will focus on:

* Your **approach and reasoning**,
* The **design choices** you made,
* Your **suggestions for improvement** and **potential real-world applications**.

We are **not evaluating the absolute performance** of your model, but rather your **methodology**, **clarity of explanation**, and **ability to present your work**.

#### ***📦 Provided Materials***

You will receive:

* A **fictional knowledge base** (/docs) containing **10 short .md files**,
* A list of **natural-language questions** (questions.json),
* A **context token limit** of **1024 tokens per question**.

#### ***🧠 Your Tasks***

1. **Chunking & Retrieval**
   1. Split the knowledge base into **meaningful chunks**.
   2. Design a **retrieval and scoring strategy** that ensures only the most relevant text is included within the token budget.
2. **Answer Generation**
   1. Use a **small local LLM** (1.7B–2B parameters) to generate answers.
   2. Ensure your pipeline effectively integrates with the retrieval component.
3. **Evaluation & Justification**
   1. For each question, decide **what context to show** and **why**.
   2. Explain **how and why** your approach works (or where it might fail).
   3. Optionally, suggest metrics or manual checks to evaluate performance.

#### ***📁 Deliverables***

Your submission should include:

1. **GitHub Repository**
   1. Clean, well-documented code
   2. Clear **execution instructions**
   3. Must be shared **at least one day before the interview**
2. **Presentation (PowerPoint or PDF)**
   1. Your **approach and design rationale**
   2. **Results**, **model choices**, and **key challenges**
   3. **Areas for improvement** and future development ideas
   4. *(Bonus)* A **deployment architecture diagram** if you were to productionize the solution
