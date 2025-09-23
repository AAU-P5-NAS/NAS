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
                source ${VENV_DIR}/bin/activate
                export UV_INSTALL_DIR=${VENV_DIR}/bin
                curl -LsSf https://astral.sh/uv/install.sh | sh
                uv --version
                """
            }
        }

        stage('Install Dependencies') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                uv pip install --upgrade pip
                uv pip install -e .
                """
            }
        }

        stage('Run Tests') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                pytest tests/  # adjust path if needed
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
