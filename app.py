import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI FoodRescue",
    page_icon="🍱",
    layout="centered"
)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_ID = "asadullahkhan27/AI-FoodRescue"
MODEL_FILENAME = "foodrescue_model.pth"


# =========================================================
# TITLE
# =========================================================

st.title("🍱 AI FoodRescue")
st.subheader("Intelligent Food Waste Reduction & Redistribution System")

st.write(
    """
    AI FoodRescue analyzes the visible condition of food together with
    storage information to provide an experimental rescue recommendation.
    """
)

st.info(
    "⚠️ This system is decision support only. "
    "A food image cannot reliably detect all pathogens, toxins, "
    "or microbiological hazards."
)


# =========================================================
# LOAD MODEL FROM HUGGING FACE
# =========================================================

@st.cache_resource
def load_model():

    # Download model from Hugging Face
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILENAME
    )

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    # Get class names
    classes = checkpoint["classes"]

    # Create MobileNetV2
    model = models.mobilenet_v2(
        weights=None
    )

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, classes


# =========================================================
# MODEL INITIALIZATION
# =========================================================

try:

    model, classes = load_model()

    st.success(
        "✅ AI FoodRescue model connected successfully!"
    )

except Exception as e:

    model = None
    classes = None

    st.error(
        "❌ Unable to load the Hugging Face model."
    )

    st.code(
        str(e),
        language="text"
    )

    st.warning(
        "Please check your Hugging Face repository, "
        "model filename, checkpoint format, and requirements.txt."
    )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# FOOD IMAGE UPLOAD
# =========================================================

st.header("📸 Food Analysis")

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# STORAGE INFORMATION
# =========================================================

st.header("📦 Storage Information")

storage_duration = st.number_input(
    "How many hours has the food been stored?",
    min_value=0,
    max_value=240,
    value=2,
    step=1
)


storage_location = st.selectbox(
    "Where was the food stored?",
    [
        "Refrigerator",
        "Freezer",
        "Room Temperature",
        "Unknown"
    ]
)


temperature = st.number_input(
    "Approximate storage temperature (°C)",
    min_value=-30.0,
    max_value=60.0,
    value=5.0,
    step=1.0
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Food Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze Food",
        use_container_width=True
    ):

        if model is None:

            st.error(
                "❌ Model is not available. "
                "Please fix the Hugging Face model connection first."
            )

            st.stop()


        # =================================================
        # PREPROCESS IMAGE
        # =================================================

        input_tensor = transform(image).unsqueeze(0)


        # =================================================
        # MODEL PREDICTION
        # =================================================

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )


        confidence = confidence.item()
        predicted_index = predicted_index.item()

        predicted_class = classes[predicted_index]


        # =================================================
        # CLASS PARSING
        # =================================================

        class_name = predicted_class.lower().replace(
            " ",
            "_"
        )

        food_type = "Unknown"
        condition = "Unknown"


        # Expected format:
        # bread_acceptable
        # bread_deteriorating
        # bread_spoiled

        if "_" in class_name:

            parts = class_name.rsplit(
                "_",
                1
            )

            food_type = parts[0]
            condition = parts[1]


        # =================================================
        # DISPLAY AI RESULT
        # =================================================

        st.header("🤖 AI Analysis Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Food Type",
                food_type.replace(
                    "_",
                    " "
                ).title()
            )

        with col2:

            st.metric(
                "Condition",
                condition.replace(
                    "_",
                    " "
                ).title()
            )


        st.write(
            f"**AI Confidence:** {confidence * 100:.2f}%"
        )

        st.progress(
            min(confidence, 1.0)
        )


        # =================================================
        # DECISION ENGINE
        # =================================================

        decision = ""
        recommendation = ""


        # -----------------------------------------------
        # LOW CONFIDENCE
        # -----------------------------------------------

        if confidence < 0.60:

            decision = "🔎 FURTHER ASSESSMENT"

            recommendation = (
                "The AI confidence is low. "
                "Do not rely on this prediction alone. "
                "Perform a manual inspection before consuming "
                "or redistributing the food."
            )


        # -----------------------------------------------
        # SPOILED
        # -----------------------------------------------

        elif condition == "spoiled":

            decision = "🚫 DISCARD"

            recommendation = (
                "The AI detected visible characteristics "
                "associated with spoiled food. "
                "Do not redistribute or consume it."
            )


        # -----------------------------------------------
        # UNKNOWN STORAGE
        # -----------------------------------------------

        elif storage_location == "Unknown":

            decision = "🔎 FURTHER ASSESSMENT"

            recommendation = (
                "Storage conditions are unknown. "
                "Manual food-safety assessment is recommended "
                "before making a rescue decision."
            )


        # -----------------------------------------------
        # HIGH TEMPERATURE
        # -----------------------------------------------

        elif temperature >= 30:

            decision = "🔎 FURTHER ASSESSMENT"

            recommendation = (
                "The reported storage temperature is high. "
                "The food should undergo additional safety "
                "assessment before consumption or redistribution."
            )


        # -----------------------------------------------
        # DETERIORATING FOOD
        # -----------------------------------------------

        elif condition == "deteriorating":

            if storage_duration >= 24:

                decision = "🔎 FURTHER ASSESSMENT"

                recommendation = (
                    "The food shows signs of deterioration and "
                    "has been stored for an extended period. "
                    "Manual inspection is strongly recommended."
                )

            elif storage_duration >= 12:

                decision = "🔎 FURTHER ASSESSMENT"

                recommendation = (
                    "The food may be deteriorating. "
                    "Check appearance, smell, texture, "
                    "and storage history before using."
                )

            else:

                decision = "⚠️ USE SOON"

                recommendation = (
                    "The food shows early signs of deterioration. "
                    "If it passes manual safety checks, "
                    "consider using it soon rather than storing it longer."
                )


        # -----------------------------------------------
        # ACCEPTABLE FOOD
        # -----------------------------------------------

        elif condition == "acceptable":

            if storage_duration >= 24:

                decision = "🔎 FURTHER ASSESSMENT"

                recommendation = (
                    "Although the visible condition appears acceptable, "
                    "the storage duration is long. "
                    "Perform additional safety checks."
                )

            elif storage_duration >= 12:

                decision = "🔎 FURTHER ASSESSMENT"

                recommendation = (
                    "The food appears visually acceptable, "
                    "but storage duration warrants additional "
                    "manual assessment."
                )

            else:

                decision = "✅ POTENTIAL RESCUE / USE"

                recommendation = (
                    "The food appears visually acceptable and "
                    "the reported storage duration is relatively short. "
                    "If it passes normal smell, texture, and hygiene checks, "
                    "it may be suitable for use or redistribution."
                )


        # -----------------------------------------------
        # UNKNOWN CONDITION
        # -----------------------------------------------

        else:

            decision = "🔎 FURTHER ASSESSMENT"

            recommendation = (
                "The food condition could not be confidently interpreted. "
                "Manual assessment is required."
            )


        # =================================================
        # FINAL DECISION
        # =================================================

        st.header("🎯 FoodRescue Decision")

        if "DISCARD" in decision:

            st.error(
                f"### {decision}"
            )

        elif "FURTHER" in decision:

            st.warning(
                f"### {decision}"
            )

        elif "USE SOON" in decision:

            st.warning(
                f"### {decision}"
            )

        else:

            st.success(
                f"### {decision}"
            )


        # =================================================
        # RECOMMENDATION
        # =================================================

        st.subheader("💡 Recommendation")

        st.write(
            recommendation
        )


        # =================================================
        # STORAGE SUMMARY
        # =================================================

        st.subheader("📋 Analysis Summary")

        summary = f"""
**Food Type:** {food_type.replace("_", " ").title()}

**Detected Condition:** {condition.replace("_", " ").title()}

**AI Confidence:** {confidence * 100:.2f}%

**Storage Duration:** {storage_duration} hours

**Storage Location:** {storage_location}

**Temperature:** {temperature} °C

**Final Decision:** {decision}
"""

        st.markdown(
            summary
        )


        # =================================================
        # SAFETY CHECKLIST
        # =================================================

        st.subheader("🛡️ Manual Safety Checklist")

        st.checkbox(
            "No unusual or foul smell",
            key="smell_check"
        )

        st.checkbox(
            "No visible mold or abnormal growth",
            key="mold_check"
        )

        st.checkbox(
            "No unusual texture or sliminess",
            key="texture_check"
        )

        st.checkbox(
            "Storage container was clean and covered",
            key="container_check"
        )

        st.checkbox(
            "Food was handled hygienically",
            key="hygiene_check"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🍱 AI FoodRescue | Intelligent Food Waste Reduction & Redistribution"
)

st.caption(
    "⚠️ Experimental decision-support system. "
    "AI predictions should not replace professional food-safety guidance."
)
