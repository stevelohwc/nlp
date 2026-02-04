---
skill: deploy-classifier
description: Deploy the text classification model as a standalone web application
---

# Deploy Classification Model

Deploy the trained text classification model as a standalone webpage.

## Steps

1. Check if deployment script exists at Code/text_classification/deploy.py
2. Verify trained model exists in Code/models/
3. Check if virtual environment is activated
4. Check web framework dependencies (Flask/Django/Streamlit)
5. Run deployment script
6. Display the local URL where the web app is accessible
7. Keep the server running or run in background as requested

## Expected Behavior

- Load the best trained model
- Start web server on localhost
- Provide interface for text input and classification
- Display prediction results with confidence scores
- Keep server running until manually stopped

## Notes

- Default port may vary by framework (Flask: 5000, Streamlit: 8501)
- Accept optional arguments for port, host, or model selection
- Provide clear instructions for accessing the web interface
