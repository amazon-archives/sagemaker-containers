#!/usr/bin/env bash

curl -X POST http://localhost:8080/invocations -d '[[1.0,2.0,5.0,9.0]]' -H "Content-Type: application/json" -H "Accept: application/json"
