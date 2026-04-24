# ================================================
# Product Launch Success Predictor
# Final Year Project — Desktop Application
# Built with Python + Flet + Random Forest
# ================================================

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import flet as ft
import joblib
import numpy as np
import os

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
model = joblib.load(MODEL_PATH)

# Encoding maps — must match Day 2 training exactly
COMPETITION_MAP = {'Low': 2, 'Medium': 1, 'High': 0}
CATEGORY_MAP    = {'Electronics': 0, 'Clothing': 1, 'Food': 2, 'Health': 3}
TIMING_MAP      = {'Festival': 1, 'Normal': 0}
REGION_MAP      = {'Urban': 1, 'Rural': 0}

def encode_inputs(price, budget, competition, category, timing, region):
    return np.array([[
        int(price),
        int(budget),
        COMPETITION_MAP[competition],
        CATEGORY_MAP[category],
        TIMING_MAP[timing],
        REGION_MAP[region]
    ]])

def main(page: ft.Page):
    page.title        = "Product Launch Success Predictor"
    page.window.width  = 560
    page.window.height = 700
    page.padding      = 30
    page.bgcolor      = ft.Colors.WHITE
    page.scroll       = ft.ScrollMode.AUTO

    # ── Input fields ──────────────────────────────
    price = ft.TextField(
        label="Price (₹)",
        hint_text="e.g. 8000",
        width=220,
        keyboard_type=ft.KeyboardType.NUMBER,
        border_color=ft.Colors.BLUE_200
    )
    budget = ft.TextField(
        label="Marketing Budget (₹)",
        hint_text="e.g. 250000",
        width=220,
        keyboard_type=ft.KeyboardType.NUMBER,
        border_color=ft.Colors.BLUE_200
    )
    competition = ft.Dropdown(
        label="Competition Level",
        width=220,
        options=[
            ft.dropdown.Option("Low"),
            ft.dropdown.Option("Medium"),
            ft.dropdown.Option("High"),
        ]
    )
    category = ft.Dropdown(
        label="Product Category",
        width=220,
        options=[
            ft.dropdown.Option("Electronics"),
            ft.dropdown.Option("Clothing"),
            ft.dropdown.Option("Food"),
            ft.dropdown.Option("Health"),
        ]
    )
    timing = ft.Dropdown(
        label="Launch Timing",
        width=220,
        options=[
            ft.dropdown.Option("Festival"),
            ft.dropdown.Option("Normal"),
        ]
    )
    region = ft.Dropdown(
        label="Target Region",
        width=220,
        options=[
            ft.dropdown.Option("Urban"),
            ft.dropdown.Option("Rural"),
        ]
    )

    # ── Result display ────────────────────────────
    result_text = ft.Text(
        value="",
        size=22,
        weight=ft.FontWeight.BOLD,
        text_align=ft.TextAlign.CENTER
    )
    confidence_text = ft.Text(
        value="",
        size=14,
        color=ft.Colors.GREY_600,
        text_align=ft.TextAlign.CENTER
    )
    result_container = ft.Container(
        content=ft.Column(
            [result_text, confidence_text],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=4
        ),
        padding=16,
        border_radius=10,
        visible=False
    )

    # ── Predict logic ─────────────────────────────
    def predict(e):
        # Validate all fields filled
        if not all([price.value, budget.value,
                    competition.value, category.value,
                    timing.value, region.value]):
            result_text.value             = "Please fill all fields"
            result_text.color             = ft.Colors.ORANGE_700
            confidence_text.value         = ""
            result_container.bgcolor      = ft.Colors.ORANGE_50
            result_container.visible      = True
            page.update()
            return

        # Validate numeric fields
        try:
            p = int(price.value)
            b = int(budget.value)
            if p <= 0 or b <= 0:
                raise ValueError
        except ValueError:
            result_text.value             = "Price and Budget must be positive numbers"
            result_text.color             = ft.Colors.ORANGE_700
            confidence_text.value         = ""
            result_container.bgcolor      = ft.Colors.ORANGE_50
            result_container.visible      = True
            page.update()
            return

        # Encode and predict
        try:
            X             = encode_inputs(
                                price.value, budget.value,
                                competition.value, category.value,
                                timing.value, region.value
                            )
            prediction    = model.predict(X)[0]
            probs         = model.predict_proba(X)[0]
            confidence    = probs[prediction] * 100

            if prediction == 1:
                result_text.value         = "SUCCESS"
                result_text.color         = ft.Colors.GREEN_700
                confidence_text.value     = f"Confidence: {confidence:.1f}%"
                confidence_text.color     = ft.Colors.GREEN_600
                result_container.bgcolor  = ft.Colors.GREEN_50
            else:
                result_text.value         = "FAILURE"
                result_text.color         = ft.Colors.RED_700
                confidence_text.value     = f"Confidence: {confidence:.1f}%"
                confidence_text.color     = ft.Colors.RED_600
                result_container.bgcolor  = ft.Colors.RED_50

            result_container.visible = True

        except Exception as ex:
            result_text.value        = f"Error: {str(ex)}"
            result_text.color        = ft.Colors.RED_700
            confidence_text.value    = ""
            result_container.visible = True

        page.update()

    # ── Button (new API) ──────────────────────────
    predict_btn = ft.FilledButton(
        text="Predict Launch Outcome",
        width=260,
        height=44,
        on_click=predict
    )

    # ── Page layout ───────────────────────────────
    page.add(
        ft.Text(
            "Product Launch Success Predictor",
            size=20,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.BLUE_800
        ),
        ft.Text(
            "Powered by Random Forest — Final Year Project",
            size=12,
            color=ft.Colors.GREY_600
        ),
        ft.Divider(height=20),

        ft.Text("Product Details",
                size=13, weight=ft.FontWeight.W_500,
                color=ft.Colors.GREY_700),
        ft.Row([price, budget], spacing=16),
        ft.Row([competition, category], spacing=16),

        ft.Divider(height=10),
        ft.Text("Market & Launch Conditions",
                size=13, weight=ft.FontWeight.W_500,
                color=ft.Colors.GREY_700),
        ft.Row([timing, region], spacing=16),

        ft.Divider(height=20),
        ft.Row(
            [predict_btn],
            alignment=ft.MainAxisAlignment.CENTER
        ),
        ft.Divider(height=10),
        ft.Row(
            [result_container],
            alignment=ft.MainAxisAlignment.CENTER
        ),
    )

ft.run(main)