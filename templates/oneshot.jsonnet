local targets = import 'all_targets.jsonnet';

function(test) targets[test].oneshotJob
