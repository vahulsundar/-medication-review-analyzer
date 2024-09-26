st.title("Medication Review Analyzer")

st.write("Enter your medication review below to analyze its category:")

user_input = st.text_area("Review Text", height=200)

if st.button("Analyze"):
    if user_input.strip():
        with st.spinner('Analyzing...'):
            # Preprocess the input text using Sentence Transformer
            input_embedding = sentence_model.encode([user_input])

            # Make prediction using the XGBoost model
            prediction = model.predict(input_embedding)
            predicted_category = label_encoder.inverse_transform(prediction)[0]

            # Get prediction probabilities
            prediction_proba = model.predict_proba(input_embedding)
            confidence = np.max(prediction_proba) * 100

            # Display the results
            st.success(f"**Predicted Category:** {predicted_category}")
            st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("Please enter a review text.")
