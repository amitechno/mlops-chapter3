Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories

* Create a directory for the project and initialize Git and DVC.
   * As you work on the code, continually commit changes. Trained models you want to keep must be committed to DVC.
* Connect your local Git repository to GitHub.

## Set up S3

* In your CLI environment install the<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html" target="_blank"> AWS CLI tool</a>.
* In the navigation bar in the Udacity classroom select **Open AWS Gateway** and then click **Open AWS Console**. You will not need the AWS Access Key ID or Secret Access Key provided here.
* From the Services drop down select S3 and then click Create bucket.
* Give your bucket a name, the rest of the options can remain at their default.

To use your new S3 bucket from the AWS CLI you will need to create an IAM user with the appropriate permissions. The full instructions can be found <a href="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console" target="_blank">here</a>, what follows is a paraphrasing:

* Sign in to the IAM console <a href="https://console.aws.amazon.com/iam/" target="_blank">here</a> or from the Services drop down on the upper navigation bar.
* In the left navigation bar select **Users**, then choose **Add user**.
* Give the user a name and select **Programmatic access**.
* In the permissions selector, search for S3 and give it **AmazonS3FullAccess**
* Tags are optional and can be skipped.
* After reviewing your choices, click create user. 
* Configure your AWS CLI to use the Access key ID and Secret Access key.

## GitHub Actions

* Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
   * Make sure you set up the GitHub Action to have the same version of Python as you used in development.
* Add your <a href="https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions" target="_blank">AWS credentials to the Action</a>.
* Set up <a href="https://github.com/iterative/setup-dvc" target="_blank">DVC in the action</a> and specify a command to `dvc pull`.

## Data

* Download census.csv from the data folder in the starter repository.
   * Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.
* Create a remote DVC remote pointing to your S3 bucket and commit the data.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.
* Commit this modified data to DVC under a new name (we often want to keep the raw data untouched but then can keep updating the cooked version).

## Model

* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
   * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

## API Creation

* Create a RESTful API using FastAPI this must implement:
   * GET on the root giving a welcome message.
   * POST that does model inference.
   * Type hinting must be used.
   * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment

* Sign up for an account on Render: Go to https://render.com/ and sign up for a free account if you don't have one already.
* Create a New Web Service: Once you have signed in to your Render account, click on "Create a new web service" to get started.
* Choose a Repository: Select the repository that contains your project code. Render will automatically detect the language and framework used in your project.
* Configure the Service: Configure the web service settings according to your project requirements. This will include choosing a name for the service, selecting a branch to deploy, and specifying the build command and port to run your application.
* Deploy Your Project: Click on "Create Web Service" to start the deployment process. Render will automatically build and deploy your project.
* Access Your App: Once the deployment is complete, Render will provide you with a URL where your app is accessible.
* Verify Your App: Visit the provided URL in your web browser to verify that your app is running correctly.
* Continuous Deployment: Render supports continuous deployment, which means that every time you push changes to your project's repository, Render will automatically build and deploy the latest version of your app.
* Custom Domains (Optional): If you want to use a custom domain for your app, you can configure it in the Render dashboard.
