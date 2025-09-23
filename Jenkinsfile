pipeline {
    agent any

    environment {
        VENV_DIR = "${WORKSPACE}/venv"
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

        stage('Install Dependencies') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                curl -LsSf https://astral.sh/uv/install.sh | sh
                export PATH=${VENV_DIR}/bin:$PATH
                uv install --upgrade
                """
            }
        }

        stage('Run Tests') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                pytest tests/  # adjust the path to your test directory if needed
                """
            }
        }

        stage('Build') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                uv build  # or replace with your build command if different
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
