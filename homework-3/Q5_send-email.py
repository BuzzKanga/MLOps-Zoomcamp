from prefect import flow
from prefect_email import EmailServerCredentials, email_send_message

@flow
def example_email_send_message_flow():
    email_server_credentials = EmailServerCredentials.load("email-server-credentials")

    subject = email_send_message(
        email_server_credentials=email_server_credentials,
        subject="Example Flow Notification using Gmail",
        msg="This proves email_send_message works!",
        email_to="stevekaren@gmail.com",
    )
    return subject


example_email_send_message_flow()

