local common = import '../jax/common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';


{
  PaxTest:: common.JaxTest {
    frameworkPrefix: 'mp-pax',
    paxmlVersion:: 'stable'
    testScript:: |||
	set -x
	set -u
	set -e
	# .bash_logout sometimes causes a spurious bad exit code, remove it.
      	rm .bash_logout

	# Install stable praxis and paxml
	pip install praxis
	pip install paxml

	# paxml version
	paxml_version = `pip show paxml`
	echo "$paxml_version"
    |||,
  }
}

