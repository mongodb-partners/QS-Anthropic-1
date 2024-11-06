# Customer service travel advisory application with AWS, Anthropic and MongoDB 
## Technical Overview
The use case in focus involves a travel advisory service designed for end users. This application enables users to ask questions related to travel itineraries and planning, offering personalized recommendations based on individual preferences.
At the core of the architecture is MongoDB Atlas, which functions both as the primary content store and the vector database for the system. The architecture exhibits agent-based behavior, where the application intelligently leverages multiple agents to process user requests and deliver the most accurate and relevant responses. This dynamic approach ensures that the system adapts to various types of user inquiries, providing tailored solutions.
Goal: To assist the customer in planning a vacation by providing personalized recommendations based on their preferences.

![image](https://github.com/user-attachments/assets/ba237d6e-9df7-4200-ac2c-52b97e556409)

**Figure:** Architecture diagram illustrating the integration of AWS, Anthropic, and MongoDB for building an AI-enhanced end-to-end application.


Clone the repo using git clone : https://github.com/mongodb-partners/QS-Anthropic-1.git



## Components:
### MongoDB Atlas  
MongoDB Atlas will serve as the primary database, responsible for securely storing structured and unstructured trip recommendation data. Additionally, it will house embeddings and chat history that are essential for advanced search and recommendation features, enabling more relevant and personalized travel suggestions. 
1. Create MongoDB Atlas on AWS.
2. Load the documents from the file *anthropic-travel-agency.trip_recommendations.csv* to MongoDB Atlas either using [MongoDB Compass](https://www.mongodb.com/docs/compass/current/import-export/#import-data-into-a-collection).




### Anthropic on AWS Bedrock  
We are leveraging Anthropic's AI models to enhance the quality and richness of the data stored in MongoDB Atlas. These models will provide deeper insights and smarter recommendations by analyzing patterns within the trip data and the stored embeddings. AWS Bedrock will play a critical role in hosting and orchestrating the Anthropic models and Titan embedding models, providing a robust foundation for model management and execution. 
1. [Enable](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html) Anthropic Claude and Titan models on AWS Bedrock.

### AWS secret manager
Copy the Atlas connection string for your MongoDB Atlas collection to the AWS secret Manager
1. Open AWS secret Manger and click on Store new secret.
2. Select Other type of secret and plaintext in the Key/value pair.
3. Paste the MongoDB URI instead of JSON and update the username and password and click on Next.
4. Provide name **workshop/atlas_secret**
5. Click on next and store the secret.
   

### Lambda 
AWS Lambda will be utilized to deploy and run the backend application, offering a serverless architecture that automatically scales based on demand. The backend will handle various business logic, from processing requests to interacting with MongoDB Atlas and AWS Bedrock. Lambda's event-driven model reduces infrastructure management overhead, enabling efficient, cost-effective, and flexible application deployment.
1. To Deploy the Application on AWS Lambda follow below steps
   * [Install](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) SAM CLI.
   * From the root folder of the cloned repository run
   ``` sam build ```
   ``` sam deploy```
2. Navigate to Lambda on AWS and verify that the function is created.

### Lex
AWS Lex will be employed to create an intuitive user interface, offering natural language processing capabilities to enhance user interaction.
1. To test out the application Open Amazon Lex
2. Click on Create Bot and provide a name for your Bot
3. Select Create a role with basic Amazon Lex permissions.
4. Select No for Children’s Online Privacy Protection Act (COPPA) and Click on Next and then click on Done.
5. Navigate to Intents from Lex side pane for the bot created.
6. Click on Add Intent select Add empty intent with name "empty".
7. In Sample utterances type "empty" and click on Add utterances and click on Save Intent.
8. navigate to FallbackIntent and activate the Fulfillment using radio button and click on Save intent.
9. Build intent the bot using the build button on the top and once built click on Test.
10. Ask questions related to places to visit like India, Singapore, Bhutan etc.
