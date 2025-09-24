pipeline {
    agent any

    environment {
        UV_BIN = "${HOME}/.local/bin"
        PATH = "${UV_BIN}:${env.PATH}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install uv') {
            steps {
                sh """
                curl -LsSf https://astral.sh/uv/install.sh | sh
                uv --version
                """
            }
        }

        stage('Install Dependencies') {
            steps {
                sh """
                uv sync
                """
            }
        }

        stage('Run Tests') {
            steps {
                sh """
                uv run pytest
                """
            }
        }

        stage('Build') {
            steps {
                sh """
                uv build
                """
            }
        }
    }

    post {
        success {
            echo "Pipeline succeeded."
        }
        failure {
            echo "Pipeline failed."
        }
    }
}
