// Jenkinsfile — Experiment 3: Jenkins → Azure ML Training

pipeline {
    agent any

    environment {
        RESOURCE_GROUP  = 'rg-mlops-exp'
        WORKSPACE_NAME  = 'mlops-workspace'
        EXPERIMENT_NAME = 'placement-prediction'
        COMPUTE_NAME    = 'cpu-cluster'
        ACR_NAME        = 'placementmlops.azurecr.io'
    }

    stages {

        stage('Checkout') {
            steps {
                echo '📥 Checking out code from GitHub...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo '🔧 Installing Python packages...'
                bat '"C:\\Users\\inYodreamzz\\anaconda3\\envs\\mlops\\python.exe" -m pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                echo '🧠 Training placement prediction model...'
                bat '"C:\\Users\\inYodreamzz\\anaconda3\\envs\\mlops\\python.exe" train.py'
            }
        }

        stage('Submit Azure ML Job') {
            steps {
                echo '☁️ Submitting training job to Azure ML...'
                withCredentials([
                    string(credentialsId: 'AZURE_CLIENT_ID',       variable: 'AZ_CLIENT_ID'),
                    string(credentialsId: 'AZURE_CLIENT_SECRET',    variable: 'AZ_CLIENT_SECRET'),
                    string(credentialsId: 'AZURE_TENANT_ID',        variable: 'AZ_TENANT_ID'),
                    string(credentialsId: 'AZURE_SUBSCRIPTION_ID',  variable: 'AZ_SUB_ID'),
                ]) {
                    bat """
                        "C:\\Users\\inYodreamzz\\anaconda3\\envs\\mlops\\python.exe" azure_ml_job.py ^
                            --client-id       %AZ_CLIENT_ID%     ^
                            --client-secret   %AZ_CLIENT_SECRET% ^
                            --tenant-id       %AZ_TENANT_ID%     ^
                            --subscription-id %AZ_SUB_ID%        ^
                            --resource-group  %RESOURCE_GROUP%   ^
                            --workspace       %WORKSPACE_NAME%   ^
                            --experiment      %EXPERIMENT_NAME%  ^
                            --compute         %COMPUTE_NAME%
                    """
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                bat 'docker build -t placement-api .'
            }
        }

        stage('Push to Azure Container Registry') {
            steps {
                echo '📦 Pushing image to Azure Container Registry...'
                withCredentials([
                    string(credentialsId: 'AZURE_CLIENT_ID',      variable: 'AZ_CLIENT_ID'),
                    string(credentialsId: 'AZURE_CLIENT_SECRET',  variable: 'AZ_CLIENT_SECRET'),
                ]) {
                    bat """
                        docker login %ACR_NAME% -u %AZ_CLIENT_ID% -p %AZ_CLIENT_SECRET%
                        docker tag placement-api %ACR_NAME%/placement-api:latest
                        docker push %ACR_NAME%/placement-api:latest
                    """
                }
            }
        }

    }

    post {
        success {
            echo '🎉 Pipeline complete!'
            echo '   View results: https://ml.azure.com'
        }
        failure {
            echo '❌ Pipeline failed. Check logs above.'
        }
    }
}
