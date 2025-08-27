# event_sourcing
An event sourcing pattern for storing records.

## Description

Event sourcing pattern is an approach to storing records that minimizes repetition. This pattern stores only the delta (the difference) for each individual item in the dataset. This approach allows to reduce the volume of data by filtering out repeated rows. In order to track changes that occur to the item, it needs to have a unique identifier (item key). Every time a new data snapshot comes in, only the changes between the new and the previous data snapshots are recorded based on the unique identifiers.

### Types of changes

**Created**
- If an item key appears for the first time, it is stored as "created"
- If an item key was removed and appeared again, it is also stored as "created"

**Modified**
- If changes occur to any values in the fields for a given key, the record is labelled as "modified"
- If fields/columns were added or removed, a record is also stored as "modified"

**Removed**
- If the new dataset no longer contains a key, a copy of the item from the previous record will be recorded with a "removed" label