use super::Store;
use std::collections::HashSet;

pub fn prune(store: &mut Store, extra: impl IntoIterator<Item = String>) {
    let mut protected: HashSet<String> = extra.into_iter().collect();
    for id in [&store.active, &store.latest_job].into_iter().flatten() {
        if let Some(job) = store.jobs.get(id) {
            if let Some(source) = job["source_id"].as_str() {
                protected.insert(source.to_owned());
            }
            for role in ["clean", "result"] {
                if let Some(id) = job[role]["id"].as_str() {
                    protected.insert(id.to_owned());
                }
            }
        }
    }
    for client in store.restore_clients.values() {
        protected.extend(client.assets.iter().cloned());
        protected.extend(client.result.iter().cloned());
    }
    store.assets.prune(&protected);
}
