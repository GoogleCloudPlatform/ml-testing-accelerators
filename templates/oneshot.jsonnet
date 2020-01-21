local targets = import 'targets.jsonnet';

function(test) targets[test].oneshot_job
