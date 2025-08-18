
# Disaster Recovery Plan - Lunar Habitat RL Suite

## Recovery Time Objectives (RTO)
- Critical services: 15 minutes
- Full service restoration: 60 minutes
- Data loss (RPO): 1 hour maximum

## Recovery Procedures

### 1. Regional Failover
```bash
# Automatic DNS failover via health checks
# Manual failover if needed:
kubectl config use-context eu-west-1
kubectl apply -f deployment/kubernetes/
```

### 2. Database Recovery
```bash
# Restore from latest backup
velero restore create --from-backup lunar-habitat-rl-backup-20240101-020000
```

### 3. Application Recovery
```bash
# Scale up in secondary region
kubectl scale deployment lunar-habitat-rl --replicas=5 -n lunar-habitat-rl
```

### 4. Data Synchronization
```bash
# Sync data between regions
aws s3 sync s3://lunar-habitat-rl-us-east-1/ s3://lunar-habitat-rl-eu-west-1/
```

## Monitoring and Alerting
- Health checks every 30 seconds
- Alert on 3 consecutive failures
- Automatic failover for critical services
- Manual approval for data restoration

## Testing Schedule
- Monthly failover testing
- Quarterly full DR drill
- Annual compliance audit

## Communication Plan
- Incident commander: DevOps Lead
- Stakeholder notifications via PagerDuty
- Status page updates via automated system
