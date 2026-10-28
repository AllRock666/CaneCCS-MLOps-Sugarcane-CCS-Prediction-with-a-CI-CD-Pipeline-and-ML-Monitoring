import jenkins.model.*
import hudson.security.*

def instance = Jenkins.getInstance()

// Disable security setup wizard
instance.setInstallState(jenkins.install.InstallState.INITIAL_SETUP_COMPLETED)

// Set security realm and authorization strategy allowing all
def realm = new HudsonPrivateSecurityRealm(false)
realm.createAccount("admin", "admin")
instance.setSecurityRealm(realm)

def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
strategy.setAllowAnonymousRead(true)
instance.setAuthorizationStrategy(strategy)

instance.save()
println("Jenkins security configured: admin/admin with anonymous read")
