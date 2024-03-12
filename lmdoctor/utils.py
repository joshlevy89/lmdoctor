import logging

def format_prompt(prompt, user_tag, assistant_tag):
    template_str = '{user_tag}{prompt}{assistant_tag}'
    prompt = template_str.format(user_tag=user_tag, prompt=prompt, assistant_tag=assistant_tag)
    return prompt


def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Check if handlers are already attached
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(levelname)s - %(message)s')

        # Create a handler that writes INFO logs to the console (stdout)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

