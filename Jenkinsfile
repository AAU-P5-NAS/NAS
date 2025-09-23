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
                curl -LsSf https://astral.sh/uv/install.sh | sh
                uv install --upgrade
                uv install -r requirements.txt
                uv sync
                """
            }
        }

        stage('Test') {
            steps {
                sh """
                source ${VENV_DIR}/bin/activate
                pytest --junitxml=results.xml
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
            junit 'results.xml'
            archiveArtifacts artifacts: 'dist/*', allowEmptyArchive: true
        }
    }
}
