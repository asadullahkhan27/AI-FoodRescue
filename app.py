import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI FoodRescue",
    page_icon="🍱",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# After uploading your trained model to Hugging Face,
# change this to:
#
# MODEL_ID = "YOUR_USERNAME/AI-FoodRescue"
#
MODEL_ID = "asadullahkhan27/AI-FoodRescue"

MODEL_FILENAME = "foodrescue_model.pth"


# ============================================================
# HEADER
# ============================================================

st.title("🍱 AI FoodRescue")

st.subheader(
    "AI-Assisted Food Rescue & Decision Support"
)

st.write(
    """
    AI FoodRescue analyzes the visible condition of selected food
    items and combines the visual assessment with storage information
    to provide an AI-assisted recommendation.
    """
)

st.info(
    """
    ⚠️ This is a decision-support prototype.
    Image analysis cannot determine microbiological safety,
    detect all pathogens or toxins, or guarantee that food is
    safe to eat.
    """
)


# ============================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================

@st.cache_resource
def load_model():

    if MODEL_ID == "YOUR_USERNAME/AI-FoodRescue":
        return None, None

    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILENAME
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    classes = checkpoint["classes"]

    model = models.mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, classes


try:

    model, classes = load_model()

except Exception as e:

    model = None
    classes = None

    st.error(
        "Unable to load the Hugging Face model."
    )

    st.code(str(e))


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("1. Upload Food Image")

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Food",
        use_container_width=True
    )


    # ========================================================
    # STORAGE INFORMATION
    # ========================================================

    st.header("2. Storage Information")

    storage_duration = st.selectbox(
        "How long has the food been stored?",

        [
            "Less than 6 hours",
            "6–12 hours",
            "12–24 hours",
            "More than 24 hours",
            "Unknown"
        ]
    )


    storage = st.selectbox(
        "Where was it stored?",

        [
            "Refrigerated",
            "Room temperature",
            "Warm environment",
            "Unknown"
        ]
    )


    temperature = st.number_input(
        "Approximate storage temperature (°C)",

        min_value=0,
        max_value=60,

        value=25
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    st.header("3. AI Analysis")

    analyze = st.button(
        "🔍 Analyze Food",
        type="primary",
        use_container_width=True
    )


    if analyze:

        # ====================================================
        # CHECK MODEL
        # ====================================================

        if model is None:

            st.warning(
                """
                The Hugging Face model is not connected yet.
                First upload your trained
                `foodrescue_model.pth` model to Hugging Face
                and update MODEL_ID in app.py.
                """
            )

            st.stop()


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        input_tensor = transform(
            image
        ).unsqueeze(0)


        with torch.no_grad():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )


        confidence = confidence.item()


        predicted_class = classes[
            predicted_index.item()
        ]


        # ====================================================
        # EXTRACT FOOD + CONDITION
        # ====================================================

        parts = predicted_class.split("_")


        if len(parts) >= 2:

            condition = parts[-1]

            food_type = "_".join(
                parts[:-1]
            )

        else:

            food_type = "unknown"

            condition = "unknown"


        # ====================================================
        # DISPLAY AI RESULTS
        # ====================================================

        st.divider()

        st.header("AI Visual Analysis")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Food Category",
                food_type.capitalize()
            )


        with col2:

            st.metric(
                "Visible Condition",
                condition.capitalize()
            )


        st.write(
            f"**AI Confidence:** "
            f"{confidence * 100:.1f}%"
        )


        # ====================================================
        # DECISION ENGINE
        # ====================================================

        if confidence < 0.60:

            decision = "FURTHER ASSESSMENT"

            reason = (
                "The AI model has relatively low confidence "
                "in its visual classification."
            )


        elif condition == "spoiled":

            decision = "DISCARD"

            reason = (
                "The image was classified as visibly spoiled. "
                "The system does not recommend rescue based "
                "on visual appearance."
            )


        elif storage == "Unknown":

            decision = "FURTHER ASSESSMENT"

            reason = (
                "Storage conditions are unknown. "
                "Additional food-safety assessment is required."
            )


        elif temperature >= 30:

            decision = "FURTHER ASSESSMENT"

            reason = (
                "The reported storage temperature is elevated. "
                "Additional food-safety assessment is required."
            )


        elif condition == "deteriorating":

            if storage_duration in [
                "12–24 hours",
                "More than 24 hours"
            ]:

                decision = "FURTHER ASSESSMENT"

                reason = (
                    "Visible deterioration combined with "
                    "extended storage requires further assessment."
                )

            else:

                decision = "USE SOON"

                reason = (
                    "Visible deterioration was detected. "
                    "Prioritize use only if applicable "
                    "food-safety requirements are satisfied."
                )


        elif condition == "acceptable":

            if storage_duration in [
                "12–24 hours",
                "More than 24 hours"
            ]:

                decision = "FURTHER ASSESSMENT"

                reason = (
                    "The food appears visually acceptable, "
                    "but extended storage requires additional "
                    "assessment."
                )

            else:

                decision = "POTENTIAL RESCUE / USE"

                reason = (
                    "No obvious visible spoilage was detected. "
                    "Verify applicable food-safety requirements "
                    "before use or redistribution."
                )


        else:

            decision = "FURTHER ASSESSMENT"

            reason = (
                "The system could not confidently determine "
                "the food condition."
            )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.divider()

        st.header(
            "FoodRescue Recommendation"
        )


        if decision == "DISCARD":

            st.error(
                f"### ❌ {decision}"
            )


        elif decision == "FURTHER ASSESSMENT":

            st.warning(
                f"### ⚠️ {decision}"
            )


        elif decision == "USE SOON":

            st.warning(
                f"### ⏱️ {decision}"
            )


        else:

            st.success(
                f"### ✅ {decision}"
            )


        st.write(
            f"**Reason:** {reason}"
        )


        # ====================================================
        # ASSESSMENT SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "Assessment Summary"
        )


        st.write(
            f"**Food:** "
            f"{food_type.capitalize()}"
        )


        st.write(
            f"**Visible Condition:** "
            f"{condition.capitalize()}"
        )


        st.write(
            f"**Storage:** "
            f"{storage}"
        )


        st.write(
            f"**Duration:** "
            f"{storage_duration}"
        )


        st.write(
            f"**Temperature:** "
            f"{temperature} °C"
        )


        st.write(
            f"**AI Confidence:** "
            f"{confidence * 100:.1f}%"
        )


        st.write(
            f"**Recommendation:** "
            f"{decision}"
        )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.divider()

        st.caption(
            """
            AI FoodRescue is an experimental decision-support system.
            Image analysis cannot detect all pathogens, toxins, or
            food-safety hazards. Always follow established food-safety
            procedures and applicable regulations.
            """
        )
