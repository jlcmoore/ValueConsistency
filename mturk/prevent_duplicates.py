import argparse
import logging
import signal
import sys
import time

import boto3
import botocore.config

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('mturk_processing.log')

# Create formatters and add them to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# TODO: don't release these strings
COMPLETED_QUAL_ID_SANDBOX = "35GMP50387F60CD6JE9GMB62XBH93C"
COMPLETED_QUAL_ID = "3H3KEN1OM8X4E8AM21ZHNJSJM2EBIH"
MTURK_SANDBOX_ENDPOINT = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"

def get_mturk_client(sandbox):
    """
    Initialize and return an MTurk client.

    :param sandbox: Boolean indicating whether to use the sandbox environment.
    :return: MTurk client.
    """
    kwargs = {}
    kwargs['config'] = botocore.config.Config(
       retries = {
          'max_attempts': 10,
          'mode': 'standard'
       }
    )
    if sandbox:
        logger.debug('Added sandbox endpoint')
        kwargs['endpoint_url'] = MTURK_SANDBOX_ENDPOINT
    client = boto3.client('mturk', region_name='us-east-1', **kwargs)
    return client

def get_hits_by_name(mturk, hit_name, qualification_id=None, require_assignable=False):
    """
    Retrieve all HITs with a certain name.

    :param mturk: MTurk client.
    :param hit_name: Name of the HIT.
    :param qualification_id: Optional qualification ID to filter HITs.
    :param require_assignable: Boolean flag to require HIT status to be 'Assignable'.
    :return: List of HITs.
    """
    hits = []
    next_token = None

    while True:
        if next_token:
            response = mturk.list_hits(NextToken=next_token)
        else:
            response = mturk.list_hits()
        for hit in response['HITs']:
            logger.debug(f"Got hit title, {hit['Title']}")
            if hit['Title'] == hit_name:
                logger.debug("Equals")
                logger.debug(f"Status: {hit['HITStatus']}")                
                if require_assignable and hit['HITStatus'] != 'Assignable':
                    continue


                if qualification_id:
                    hit_details = mturk.get_hit(HITId=hit['HITId'])
                    qualifications = hit_details['HIT']['QualificationRequirements']
                    if not any(q['QualificationTypeId'] == qualification_id for q in qualifications):
                        continue

                hits.append(hit)

        next_token = response.get('NextToken')
        if not next_token:
            break

    return hits

def get_workers_with_qualification(mturk, qualification_id):
    """
    Retrieve all worker IDs with a specific qualification.

    :param mturk: MTurk client.
    :param qualification_id: Qualification ID.
    :return: List of worker IDs.
    """
    worker_ids = []
    next_token = None

    try:
        while True:
            if next_token:
                response = mturk.list_workers_with_qualification_type(
                    QualificationTypeId=qualification_id,
                    NextToken=next_token
                )
            else:
                response = mturk.list_workers_with_qualification_type(
                    QualificationTypeId=qualification_id
                )
            for qualification in response['Qualifications']:
                worker_ids.append(qualification['WorkerId'])

            next_token = response.get('NextToken')
            if not next_token:
                break
    except Exception as e:
        logger.error(f"Error retrieving workers with qualification {qualification_id}: {e}")
    return worker_ids

def process_assignments(mturk, hits, qualification_id=None):
    """
    Process assignments for given HITs, approving all instances and creating additional assignments for duplicates.

    :param mturk: MTurk client.
    :param hits: List of HITs.
    :param qualification_id: Qualification ID to assign to workers.
    """
    worker_assignment_count = {}

    for hit in hits:
        hit_id = hit['HITId']
        next_token = None

        while True:
            if next_token:
                response = mturk.list_assignments_for_hit(HITId=hit_id, NextToken=next_token)
            else:
                response = mturk.list_assignments_for_hit(HITId=hit_id)

            assignments = response['Assignments']
            for assignment in assignments:
                worker_id = assignment['WorkerId']
                assignment_id = assignment['AssignmentId']
                assignment_status = assignment['AssignmentStatus']

                if worker_id not in worker_assignment_count:
                    worker_assignment_count[worker_id] = 0

                if assignment_status == 'Submitted':
                    if worker_assignment_count[worker_id] == 0:
                        # Qualify worker
                        try:
                            mturk.associate_qualification_with_worker(
                                QualificationTypeId=qualification_id,
                                WorkerId=worker_id,
                                SendNotification=False
                            )
                            logger.info(f"Assigned qualification {qualification_id} to worker {worker_id}")
                        except Exception as e:
                            logger.error(f"Error assigning qualification to worker {worker_id}: {e}")
                    else:
                        try:
                            # Create additional assignments for the HIT
                            mturk.create_additional_assignments_for_hit(
                                HITId=hit_id,
                                NumberOfAdditionalAssignments=1
                            )
                            logger.info(f"Created additional assignment for HIT {hit_id} due to duplicate submission by worker {worker_id}")
                        except Exception as e:
                            logger.error(f"Error approving duplicate assignment {assignment_id} or creating additional assignments for HIT {hit_id}: {e}")

                    # Approve worker, regardless of it being a duplicate (need to do this for "Submitted" type)
                    try:
                        mturk.approve_assignment(AssignmentId=assignment_id)
                        logger.info(f"Approved assignment {assignment_id} for worker {worker_id}")
                    except Exception as e:
                        logger.error(f"Error approving assignment {assignment_id} for worker {worker_id}: {e}")
                
                # Add to the worker's count regardless of the assignment's status    
                worker_assignment_count[worker_id] += 1

            next_token = response.get('NextToken')
            if not next_token:
                break

def main(sandbox, title):
    """
    Main function to process assignments and prevent duplicate submissions.

    :param sandbox: Boolean indicating whether to use the sandbox environment.
    """
    if sandbox:
        complted_qual_id = COMPLETED_QUAL_ID_SANDBOX
    else:
        complted_qual_id = COMPLETED_QUAL_ID

    mturk = get_mturk_client(sandbox)
    logger.debug("got client")
    while not stop_signal:
        logger.debug("Querying for hits")
        hits = get_hits_by_name(mturk, title, qualification_id=complted_qual_id, require_assignable=False)
        process_assignments(mturk, hits, qualification_id=complted_qual_id)
        time.sleep(20)  # Poll every 20 seconds

def signal_handler(sig, frame):
    global stop_signal
    logger.info('Interrupt signal received. Exiting gracefully...')
    stop_signal = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='prevent_duplicates',
                    description='Prevents duplicate submissions. Approves Workers. Qualifies them.')
    parser.add_argument('--sandbox', action="store_true", default=False,
                        help="Whether to use the sandbox environment.")
    parser.add_argument('--title', required=True, help="The title of the HITs")
    args = parser.parse_args()

    stop_signal = False

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main(**vars(args))
