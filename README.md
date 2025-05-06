# Document Classifier with Cloud Integration

## Solution

Thank you for reviewing my solution. If you have any questions, please reach out to me at tottility@gmail.com. I would be happy to make any clarification or improve anything further.

The original solution relied on word-matching in filenames to determine document types, which is limited and error-prone for poorly named files.

I improved this by implementing a light-weight Machine Learning solution using a Linear SVM (Support Vector Machine) classifier with TF-IDF vectorization. This approach analyzes the actual content of files rather than just their names. Files are preprocessed based on their extension type, with OCR applied to image files, text extraction from PDFs and Word documents, and direct processing for text files.

For training data generation, I leveraged the OpenAI model to create synthetic documents across different types and industries. This approach is highly extensible - you can generate training data for new document types or industries by calling the `/generate_training_data` endpoint with the desired parameters. After generating new data, you can retrain the model via the `/train_model` endpoint to expand its classification capabilities.

To ensure the classifier is robust and reliable in production, I:
1. Implemented a CI/CD pipeline with GitHub Actions for automated testing and deployment
2. Deployed the service on Google Cloud Run for scalability
3. Used gunicorn as the WSGI server to handle concurrent requests
4. Added Google Cloud Storage integration for persistent model storage between deployments

Currently, the model supports these document types:
```
["invoice", "bank_statement", "receipt", "contract", "lease_agreement",
 "drivers_licence", "passport", "resume", "transcript"]
```

The system can be trained on any industry context, such as healthcare, finance, legal, education, and technology.


### Test the Deployed Application

The deployed application have the following endpoints:

**1. Generate Training Data:**
```bash
curl -X POST https://document-classifier-606021919371.us-central1.run.app/generate_training_data \
  -H "Content-Type: application/json" \
  -d '{"doc_types": ["invoice", "resume", "contract"], "samples_per_type": 10, "industry": "healthcare"}'
```

**2. Check Data Generation Status:**
```bash
curl https://document-classifier-606021919371.us-central1.run.app/data_generation_status
```

**3. Train the Model:**
```bash
curl -X POST https://document-classifier-606021919371.us-central1.run.app/train_model \
  -H "Content-Type: application/json" \
  -d '{}'
```

**4. Check Training Status:**
```bash
curl https://document-classifier-606021919371.us-central1.run.app/training_status
```

**5. Classify a Document:**
```bash
curl -X POST https://document-classifier-606021919371.us-central1.run.app/classify_file -F "file=@test_invoice.txt"
```