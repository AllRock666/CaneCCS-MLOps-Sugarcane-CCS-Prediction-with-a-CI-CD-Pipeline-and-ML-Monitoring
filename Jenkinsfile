// Declarative CI/CD pipeline for CaneCCS MLOps
// Works on Linux agents (sh) and Windows agents (bat).
def run(cmd) { if (isUnix()) { sh cmd } else { bat cmd } }

pipeline {
    agent any
    options { timestamps(); timeout(time: 20, unit: 'MINUTES') }
    environment {
        IMAGE = "canesugar-mlops"
        TAG   = "${env.BUILD_NUMBER}"
    }
    stages {
        stage('Checkout') {
            steps { checkout scm }
        }
        stage('Install deps') {
            steps {
                run 'python -m venv venv'
                script {
                    if (isUnix()) { sh 'venv/bin/pip install -r requirements.txt' }
                    else          { bat 'venv\\Scripts\\pip install -r requirements.txt' }
                }
            }
        }
        stage('Train model') {
            steps {
                script {
                    if (isUnix()) { sh 'venv/bin/python train.py' }
                    else          { bat 'venv\\Scripts\\python train.py' }
                }
            }
        }
        stage('Unit tests') {
            steps {
                script {
                    if (isUnix()) { sh 'venv/bin/python -m pytest -q --junitxml=report.xml' }
                    else          { bat 'venv\\Scripts\\python -m pytest -q --junitxml=report.xml' }
                }
            }
            post { always { junit 'report.xml' } }
        }
        stage('Docker build') {
            steps { run "docker build -t ${IMAGE}:${TAG} -t ${IMAGE}:latest ." }
        }
        stage('Deploy stack') {
            steps { run 'docker compose up -d --build' }
        }
        stage('Smoke test') {
            steps {
                script {
                    sleep 8
                    if (isUnix()) { sh 'curl -f http://localhost:5000/health' }
                    else          { bat 'curl -f http://localhost:5000/health' }
                }
            }
        }
    }
    post {
        success { echo "Build ${env.BUILD_NUMBER}: CaneCCS deployed. Grafana :3000, Prometheus :9090" }
        failure { echo "Build ${env.BUILD_NUMBER} failed - check stage logs" }
    }
}
