# Setup + Demo Procedure Using Docker Compose

This document lays out the steps to the full environment of this
project, both locally and as a demo for a public audience.

Prerequisites:
- a Docker installation
- an Ngrok account (any tier, only if you intend on setting up a
  public demo)

## Part 1: Local Setup

1. **Build the Docker Compose services**  
   First, pull/build all the images needed for this project.
   ```
   docker compose pull
   docker compose build api
   ```
2. **Pull the LLM for Ollama**  
   We then need to start the Ollama service before any of the others
   in order to configure with a large language model.
   ```
   docker compose up ollama
   docker compose exec -it ollama ollama pull llama3.1:8b
   ```
3. **Start a local instance**  
   At this point, we should be able to launch the API service and UI
   service as well. Launch them with the following command:
   ```
   docker compose up -d
   ```
4. **Sign into OpenWebUI and configure**  
   Once the previous step has loaded completely, you should be able to
   navigate to http://localhost:3000/ and be greeted with an OpenWebUI
   interface.
   - You will be prompted to create an admin account. Do this and you
     will be redirected to to the main chatbot interface.
   - Above the text box, it should say "valvoline-weather". If not,
     navigate to the Admin Panel > Settings > Connections. Make sure
     the "Manage OpenAI API Connections" field reads
     `http://api:8000/v1`.
     
At this point you should have the full project stack up and running
locally on your machine. Continue to Part 2 if you need to set up a
public demo with Ngrok. Otherwise, skip to Part 3.

## Part 2: Public Demo Setup

5. **Configure /ngrok.yml**  
   Ngrok needs your account's authtoken to in order to serve anything
   to a public URL. You can find this by signing into
   https://ngrok.com and navigating to "Your Authtoken" in the
   dashboard. After copying this value, you can use it to configure
   the Ngrok service's agent:
   - Go to this repository's root directory and copy
     `ngrok.yml.example` to `ngrok.yml`
   - In the new `ngrok.yml`, go to `authtoken: change-me` and change
     the `change-me` value to your authtoken
     - Make sure this is done in `ngrok.yml` and **not** in
       `ngrok.yml.example`. Writing to the latter will not only not
       work, it'll also expose your private token should you push
       those changes upstream
6. **Start the Ngrok service**  
   You can then start the Ngrok service:
   ```
   docker compose --profile demo up
   ```
   Once it's loaded, navigate to the status page http://localhost:4040
   and there you should find your public facing URL.
7. **Configure the OpenWebUI URL**  
   The Ngrok service will not be able to connect to the UI service
   until OpenWebUI knows about your public Ngrok URL. Navigate to the
   the OpenWebUI instance (http://localhost:3000) and do the
   following:
   - Navigate to the Admin Panel > Settings > General
   - Scroll down to where it says "WebUI URL" and paste your URL
   - Save the settings
   
   NOTE: If you're using a free Ngrok account, your public URL is not
   guaranteed to be the same each time you launch the service. It's
   recommended you check each time to make sure the WebUI URL and the
   Ngrok URL are the same.
8. **Create a guest account**  
   OpenWebUI does not allow users to interact with the models without
   signing in. Here's a workaround:
   - Navigate to Admin Panel > Users > Overview
   - Create a new user by clicking the "+" icon in the top right. Make
     sure the "Role" field is set to "User". The other fields can be
     whatever you like, but it's recommended you keep it simple for
     the people using the demo. For example:
     - Name: Guest
     - Email: guest@example.com
     - Password: pass123
   - After this, navigate to Admin Panel > Users > Groups and click on
     "Default Permissions". Scroll down to the permission that says
     "Enforce Temporary Chat" and switch it on. This will keep the
      user's conversations from being stored long-term, such that each
      user isn't able to view each other's conversations.
      - Alternatively, if you want long-term conversation storage for
		some demo users, you could create a group with said permission
		instead of modifying default permissions. Then, you can add
		the guest user to the group and leave the other users'
		permissions unchanged.
      
After completing these steps, anyone should be able to navigate to
your public Ngrok URL and interact with the project using the guest
account.

## Part 3: Routine Demo Workflow

At this point, you should be able to manage this project just like you
would any Docker Compose project. The following will set up your local
environment:
```
docker compose up -d
```

And if you want to set up a public demo:
```
docker compose --profile demo up -d
```
(Again, make sure your public URL is configured correctly in OpenWebUI
if you're using a free Ngrok account. See: Step 7)
