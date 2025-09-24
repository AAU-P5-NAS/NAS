pipeline {
    agent any

    environment {
        VENV_DIR = "${WORKSPACE}/venv"
        PATH = "${VENV_DIR}/bin:${env.PATH}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python') {
            steps {
                sh """
                python3 -m venv ${VENV_DIR}
                source ${VENV_DIR}/bin/activate
                python -m pip install --upgrade pip
                """
            }
        }

        stage('Install uv') {
            steps {
                sh """
                curl -LsSf https://astral.sh/uv/install.sh | sh
                export PATH="$HOME/.local/bin:$PATH"
                uv --version
                """
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh """
                export PATH="$HOME/.local/bin:$PATH"
                uv sync
                """
            }
        }
        
        stage('Run Tests') {
            steps {
                sh """
                export PATH="$HOME/.local/bin:$PATH"
                uv run pytest
                """
            }
        }


        stage('Build') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                uv build
                """
            }
        }
    }

    post {
        always {
            sh "rm -rf ${VENV_DIR}"
        }
        success {
            echo "Pipeline succeeded."
        }
        failure {
            echo "Pipeline failed."
        }
    }
}
