import os
import requests
import logging
from utils import fetch_wikipedia_info
from utils import escape_markdown
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    JobQueue,
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load API URL and BOT TOKEN from environment variables
API_URL = "http://127.0.0.1:8000/predict"
BOT_TOKEN = "8109572586:AAHyjG-j45IZmZUV6S6fY1OmBQ-r5Gz0GQY"
async def start(update: Update, context: CallbackContext):
    """Welcome message for the bot."""
    welcome_message = (
        "👋 *Welcome to MediMate Bot!*\n\n"
        "I can help predict possible medical conditions based on your symptoms.\n"
        "Please describe your symptoms, separating them with commas.\n\n"
        "📝 *Example:* `fever, headache, cough`"
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def get_prediction(update: Update, context: CallbackContext):
    symptoms = update.message.text.strip()
    if not symptoms:
        await update.message.reply_text("⚠️ Please enter symptoms to get a prediction.")
        return

    try:
        # Send "typing" action
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Make API request for disease prediction
        response = requests.post(API_URL, json={"symptoms": symptoms}, timeout=60)
        response.raise_for_status()
        result = response.json()
        predicted_disease = result.get("disease", "Unknown condition")
        escaped_predicted_disease = escape_markdown(predicted_disease)

        # Fetch detailed information about the disease from Wikipedia
        disease_info = fetch_wikipedia_info(predicted_disease)
        escaped_disease_info = escape_markdown(disease_info)

        # Send prediction result along with Wikipedia information
        message = (
            f"🏥 *Based on your symptoms, you might have:*\n\n"
            f"🩺 *{escaped_predicted_disease}*\n\n"
            f"📚 *Information:*\n{escaped_disease_info}\n\n"
            f"⚠️ This is not a medical diagnosis. Please consult a healthcare professional."
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    except requests.exceptions.Timeout:
        await update.message.reply_text("⏳ The server is taking too long to respond. Please try again later.")
        logger.error("API request timed out.")

    except requests.exceptions.RequestException as e:
        await update.message.reply_text("❌ Sorry, I couldn't reach the prediction service. Please try again later.")
        logger.error(f"API request failed: {str(e)}")

    except ValueError:
        await update.message.reply_text("⚠️ Error: Invalid response from the prediction server.")
        logger.error("API response was not in JSON format.")

    except Exception as e:
        await update.message.reply_text("⚠️ An error occurred. Please try again later.")
        logger.error(f"Unexpected error: {str(e)}")

async def error_handler(update: object, context: CallbackContext):
    """Handle unexpected errors."""
    logger.error(f"Update {update} caused error {context.error}")
    if update and isinstance(update, Update):
        await update.message.reply_text("⚠️ An error occurred. Please try again later.")

def main():
    """Run the bot."""
   
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )
    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, get_prediction))
    application.add_error_handler(error_handler)

    # Start the bot
    logger.info("MediMate Bot started successfully!")
    application.run_polling()

if __name__ == "__main__":
    main()
