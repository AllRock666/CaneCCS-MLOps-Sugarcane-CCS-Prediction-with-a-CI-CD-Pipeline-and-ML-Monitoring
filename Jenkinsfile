pipeline {
    agent any
    options { timestamps(); timeout(time: 20, unit: 'MINUTES') }
    environment {
        IMAGE = "canesugar-mlops"
        TAG   = "${env.BUILD_NUMBER}"
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Install dependencies') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            python3 -m venv venv
                            venv/bin/pip install --upgrade pip
                            venv/bin/pip install -r requirements.txt
                        '''
                    } else {
                        bat '''
                            python -m venv venv
                            venv\\Scripts\\pip install --upgrade pip
                            venv\\Scripts\\pip install -r requirements.txt
                        '''
                    }
                }
            }
        }
        stage('Train model') {
            steps {
                script {
                    if (isUnix()) {
                        sh 'venv/bin/python train.py'
                    } else {
                        bat 'venv\\Scripts\\python train.py'
                    }
                }
            }
        }
        stage('Unit tests') {
            steps {
                script {
                    if (isUnix()) {
                        sh 'venv/bin/python -m pytest -q --junitxml=report.xml'
                    } else {
                        bat 'venv\\Scripts\\python -m pytest -q --junitxml=report.xml'
                    }
                }
            }
            post {
                always {
                    junit testResults: 'report.xml', allowEmptyResults: true
                }
            }
        }
        stage('Docker build') {
            steps {
                script {
                    if (isUnix()) {
                        sh "docker build -t ${IMAGE}:${TAG} -t ${IMAGE}:latest ."
                    } else {
                        bat "docker build -t ${IMAGE}:${TAG} -t ${IMAGE}:latest ."
                    }
                }
            }
        }
        stage('Deploy stack') {
            steps {
                script {
                    if (isUnix()) {
                        sh 'docker compose up -d --build'
                    } else {
                        bat 'docker compose up -d --build'
                    }
                }
            }
        }
        stage('Smoke test') {
            steps {
                script {
                    sleep 8
                    if (isUnix()) {
                        sh '''
                            docker run --rm --network caneccs-pipeline_default curlimages/curl:latest -f http://cane-app:5000/health || \
                            curl -f http://host.docker.internal:5000/health || \
                            curl -f http://localhost:5000/health
                        '''
                    } else {
                        bat 'curl -f http://localhost:5000/health'
                    }
                }
            }
        }
    }
    post {
        success {
            echo "Build ${env.BUILD_NUMBER}: CaneCCS deployed. Grafana :3000, Prometheus :9090"
        }
        failure {
            echo "Build ${env.BUILD_NUMBER} failed - check stage logs"
        }
    }
}
