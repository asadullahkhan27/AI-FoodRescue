import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

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

MODEL_PATH = "foodrescue_model.pth"

CONFIDENCE_THRESHOLD = 0.60

# =========================================================

# TITLE

# =========================================================

st.title("🍱 AI FoodRescue")

st.subheader(
"Intelligent Food Waste Reduction & Redistribution System"
)

st.write(
"""
AI FoodRescue analyzes the visible condition of food
together with storage information to generate an
experimental food-rescue recommendation.
"""
)

st.info(
"⚠️ Decision-support only: an image cannot reliably "
"detect all pathogens, toxins, or microbiological hazards."
)

# =========================================================

# LOAD LOCAL .PTH MODEL

# =========================================================

@st.cache_resource
def load_model():

```
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} was not found in the project folder."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

if not isinstance(checkpoint, dict):
    raise ValueError(
        "Invalid model checkpoint format."
    )

if "model_state" not in checkpoint:
    raise KeyError(
        "The model file does not contain 'model_state'."
    )

if "classes" not in checkpoint:
    raise KeyError(
        "The model file does not contain 'classes'."
    )

classes = checkpoint["classes"]

# -----------------------------------------------------
# MobileNetV2
# -----------------------------------------------------

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
```

# =========================================================

# MODEL INITIALIZATION

# =========================================================

try:

```
model, classes = load_model()

model_available = True

st.success(
    "✅ AI FoodRescue model loaded successfully!"
)
```

except Exception as e:

```
model = None
classes = None

model_available = False

st.error(
    "❌ AI model could not be loaded."
)

st.code(
    str(e),
    language="text"
)
```

# =========================================================

# IMAGE PREPROCESSING

# =========================================================

transform = transforms.Compose([

```
transforms.Resize(
    (224, 224)
),

transforms.ToTensor(),

transforms.Normalize(
    mean=[
        0.485,
        0.456,
        0.406
    ],

    std=[
        0.229,
        0.224,
        0.225
    ]
)
```

])

# =========================================================

# FOOD IMAGE

# =========================================================

st.header("📸 Food Analysis")

uploaded_file = st.file_uploader(
"Upload a food image",
type=[
"jpg",
"jpeg",
"png"
]
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

# ANALYSIS

# =========================================================

if uploaded_file is not None:

```
try:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception:

    st.error(
        "❌ Could not read the uploaded image."
    )

    st.stop()


st.image(
    image,
    caption="Uploaded Food Image",
    use_container_width=True
)


if st.button(
    "🔍 Analyze Food",
    use_container_width=True,
    type="primary"
):

    if not model_available:

        st.error(
            "❌ Model is unavailable. "
            "Make sure foodrescue_model.pth "
            "is in the same folder as app.py."
        )

        st.stop()


    # =================================================
    # AI IMAGE ANALYSIS
    # =================================================

    with st.spinner(
        "🤖 AI is analyzing the food..."
    ):

        input_tensor = transform(
            image
        ).unsqueeze(0)

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

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

    predicted_class = classes[
        predicted_index
    ]


    # =================================================
    # PARSE FOOD CLASS
    # =================================================

    class_name = (
        str(predicted_class)
        .lower()
        .strip()
        .replace(" ", "_")
    )

    food_type = "Unknown"

    condition = "Unknown"


    # Expected:
    # bread_acceptable
    # bread_deteriorating
    # bread_spoiled

    known_conditions = [
        "acceptable",
        "deteriorating",
        "spoiled"
    ]

    for detected_condition in known_conditions:

        suffix = "_" + detected_condition

        if class_name.endswith(suffix):

            food_type = class_name[
                :-len(suffix)
            ].strip("_")

            condition = detected_condition

            break


    # =================================================
    # RESULT
    # =================================================

    st.divider()

    st.header(
        "🤖 AI Analysis Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🍱 Food Type",
            food_type.replace(
                "_",
                " "
            ).title()
        )

    with col2:

        st.metric(
            "🔬 Visible Condition",
            condition.replace(
                "_",
                " "
            ).title()
        )


    st.write(
        f"**AI Confidence: "
        f"{confidence * 100:.2f}%**"
    )

    st.progress(
        min(
            max(
                confidence,
                0.0
            ),
            1.0
        )
    )


    # =================================================
    # AI ANALYTICS
    # =================================================

    st.subheader(
        "📊 AI Analytics"
    )

    if confidence >= 0.80:

        confidence_status = (
            "🟢 High confidence"
        )

    elif confidence >= 0.60:

        confidence_status = (
            "🟡 Moderate confidence"
        )

    else:

        confidence_status = (
            "🔴 Low confidence"
        )


    st.write(
        f"**Prediction Status:** "
        f"{confidence_status}"
    )


    # =================================================
    # STORAGE ANALYTICS
    # =================================================

    if storage_location == "Refrigerator":

        storage_status = (
            "🟢 Refrigerated storage"
        )

    elif storage_location == "Freezer":

        storage_status = (
            "🟢 Frozen storage"
        )

    elif storage_location == "Room Temperature":

        storage_status = (
            "🟡 Room-temperature storage"
        )

    else:

        storage_status = (
            "🔴 Unknown storage conditions"
        )


    st.write(
        f"**Storage Analysis:** "
        f"{storage_status}"
    )

    st.write(
        f"**Storage Duration:** "
        f"{storage_duration} hours"
    )

    st.write(
        f"**Temperature:** "
        f"{temperature} °C"
    )


    # =================================================
    # DECISION ENGINE
    # =================================================

    decision = ""
    recommendation = ""


    # -------------------------------------------------
    # LOW CONFIDENCE
    # -------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        decision = (
            "🔎 FURTHER ASSESSMENT"
        )

        recommendation = (
            "The AI prediction has low confidence. "
            "Do not rely on the AI result alone. "
            "Perform a manual food-safety assessment."
        )


    # -------------------------------------------------
    # SPOILED
    # -------------------------------------------------

    elif condition == "spoiled":

        decision = (
            "🚫 DISCARD"
        )

        recommendation = (
            "The model detected visible characteristics "
            "associated with spoiled food. "
            "It should not be considered for redistribution "
            "based on this visual assessment."
        )


    # -------------------------------------------------
    # UNKNOWN STORAGE
    # -------------------------------------------------

    elif storage_location == "Unknown":

        decision = (
            "🔎 FURTHER ASSESSMENT"
        )

        recommendation = (
            "Storage history is unknown. "
            "The food requires additional manual "
            "food-safety assessment."
        )


    # -------------------------------------------------
    # HIGH TEMPERATURE
    # -------------------------------------------------

    elif temperature >= 30:

        decision = (
            "🔎 FURTHER ASSESSMENT"
        )

        recommendation = (
            "The reported storage temperature is high. "
            "Additional safety assessment is required."
        )


    # -------------------------------------------------
    # DETERIORATING
    # -------------------------------------------------

    elif condition == "deteriorating":

        if storage_duration >= 24:

            decision = (
                "🔎 FURTHER ASSESSMENT"
            )

            recommendation = (
                "The food shows visible signs of deterioration "
                "and has been stored for an extended period. "
                "Manual assessment is strongly recommended."
            )

        elif storage_duration >= 12:

            decision = (
                "🔎 FURTHER ASSESSMENT"
            )

            recommendation = (
                "The food may be deteriorating. "
                "Check smell, texture, appearance, "
                "storage history, and handling conditions."
            )

        else:

            decision = (
                "⚠️ USE SOON"
            )

            recommendation = (
                "The food shows early visible signs of "
                "deterioration. If it passes normal safety "
                "checks, consider using it soon rather than "
                "allowing further waste."
            )


    # -------------------------------------------------
    # ACCEPTABLE
    # -------------------------------------------------

    elif condition == "acceptable":

        if storage_duration >= 24:

            decision = (
                "🔎 FURTHER ASSESSMENT"
            )

            recommendation = (
                "The food appears visually acceptable, "
                "but the storage duration is long. "
                "Perform additional safety checks."
            )

        elif storage_duration >= 12:

            decision = (
                "🔎 FURTHER ASSESSMENT"
            )

            recommendation = (
                "The food appears visually acceptable, "
                "but the storage duration warrants "
                "additional assessment."
            )

        else:

            decision = (
                "✅ POTENTIAL RESCUE / USE"
            )

            recommendation = (
                "The food appears visually acceptable and "
                "has a relatively short reported storage "
                "duration. If it passes normal safety checks, "
                "it may be considered for use or redistribution."
            )


    # -------------------------------------------------
    # UNKNOWN CONDITION
    # -------------------------------------------------

    else:

        decision = (
            "🔎 FURTHER ASSESSMENT"
        )

        recommendation = (
            "The food condition could not be interpreted "
            "confidently. Manual assessment is required."
        )


    # =================================================
    # FINAL DECISION
    # =================================================

    st.divider()

    st.header(
        "🎯 FoodRescue Decision"
    )


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

    st.subheader(
        "💡 Recommendation"
    )

    st.write(
        recommendation
    )


    # =================================================
    # ANALYSIS SUMMARY
    # =================================================

    st.subheader(
        "📋 Complete Analysis"
    )

    st.markdown(
        f"""
```

**🍱 Food Type:**
{food_type.replace("_", " ").title()}

**🔬 Detected Visible Condition:**
{condition.replace("_", " ").title()}

**🤖 AI Confidence:**
{confidence * 100:.2f}%

**📦 Storage Duration:**
{storage_duration} hours

**🌡️ Temperature:**
{temperature} °C

**📍 Storage Location:**
{storage_location}

**📊 Prediction Status:**
{confidence_status}

**🎯 Final Decision:**
{decision}
"""
)

```
    # =================================================
    # MANUAL SAFETY CHECKLIST
    # =================================================

    st.subheader(
        "🛡️ Manual Safety Checklist"
    )

    smell_ok = st.checkbox(
        "No unusual or foul smell"
    )

    mold_ok = st.checkbox(
        "No visible mold or abnormal growth"
    )

    texture_ok = st.checkbox(
        "No unusual texture or sliminess"
    )

    container_ok = st.checkbox(
        "Storage container was clean and covered"
    )

    hygiene_ok = st.checkbox(
        "Food was handled hygienically"
    )


    checklist_score = sum([
        smell_ok,
        mold_ok,
        texture_ok,
        container_ok,
        hygiene_ok
    ])


    st.write(
        f"**Safety checklist completed:** "
        f"{checklist_score}/5"
    )


    if checklist_score == 5:

        st.success(
            "✅ All manual checklist items are marked."
        )

    elif checklist_score > 0:

        st.warning(
            "⚠️ Some safety checks are still incomplete."
        )

    else:

        st.info(
            "ℹ️ Complete the checklist during manual assessment."
        )


    # =================================================
    # IMPORTANT DISCLAIMER
    # =================================================

    st.warning(
        """
        ⚠️ IMPORTANT

        AI FoodRescue is an experimental decision-support
        system. Visual AI cannot determine every food-safety
        hazard, including pathogens, toxins, contamination,
        or unsafe handling history.

        Do not use this system as a substitute for professional
        food-safety guidance.
        """
    )
```

# =========================================================

# FOOTER

# =========================================================

st.divider()

st.caption(
"🍱 AI FoodRescue | Intelligent Food Waste Reduction & Redistribution"
)

st.caption(
"⚠️ Experimental AI decision-support system."
)
