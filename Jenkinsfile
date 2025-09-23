pipeline {
    agent any

    environment {
        VENV_DIR = "${WORKSPACE}/venv"
        PATH = "${VENV_DIR}/bin:${env.PATH}" // Ensure venv binaries are always on PATH
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
                uv --version  # verify installation
                """
            }
        }

        stage('Install Dependencies') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                uv install --upgrade
                """
            }
        }

        stage('Run Tests') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                pytest tests/  # adjust path to your test directory
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
